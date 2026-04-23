"""
Stateful components for the RL IR runtime.
Includes ReplayBuffer, ParameterStore, and OptimizerState.
Follows the rule of explicit mutation and no hidden global state.
"""

from typing import List, Dict, Any, Optional
import torch
import random
import threading
import torch.nn as nn


class ModelRegistry:
    """
    A central registry for named PyTorch modules.
    Allows graph nodes to reference models by name (ModelHandle) 
    instead of carrying raw Python objects.
    """

    def __init__(self):
        self._models: Dict[str, nn.Module] = {}

    def register(self, name: str, model: nn.Module) -> None:
        """Registers a model under a specific handle."""
        self._models[name] = model

    def get(self, name: str) -> nn.Module:
        """Retrieves a model by its handle."""
        if name not in self._models:
            raise KeyError(f"Model handle '{name}' not found in registry.")
        return self._models[name]


class BufferRegistry:
    """
    A central registry for named ReplayBuffers.
    Allows graph nodes to reference buffers by string handles.
    """

    def __init__(self):
        self._buffers: Dict[str, Any] = {}

    def register(self, name: str, buffer: Any) -> None:
        """Registers a buffer under a specific handle."""
        self._buffers[name] = buffer

    def get(self, name: str) -> Any:
        """Retrieves a buffer by its handle."""
        if name not in self._buffers:
            raise KeyError(f"Buffer handle '{name}' not found in registry.")
        return self._buffers[name]


class OptimizerRegistry:
    """
    A central registry for named OptimizerStates.
    Allows graph nodes to reference optimizers by handle.
    """

    def __init__(self):
        self._optimizers: Dict[str, "OptimizerState"] = {}

    def register(self, name: str, optimizer: "OptimizerState") -> None:
        """Registers an optimizer under a specific handle."""
        self._optimizers[name] = optimizer

    def get(self, name: str) -> "OptimizerState":
        """Retrieves an optimizer by its handle."""
        if name not in self._optimizers:
            raise KeyError(f"Optimizer handle '{name}' not found in registry.")
        return self._optimizers[name]


class ReplayBuffer:
    """
    A simple circular replay buffer for storing transitions.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Dict[str, torch.Tensor]] = []
        self.position = 0
        self._lock = threading.Lock()

    def add(self, transition: Dict[str, torch.Tensor]) -> None:
        """Adds a transition to the buffer."""
        new_entry = {
            k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in transition.items()
        }
        with self._lock:
            if len(self.buffer) < self.capacity:
                self.buffer.append(new_entry)
            else:
                self.buffer[self.position] = new_entry
            self.position = (self.position + 1) % self.capacity

    def clear(self) -> None:
        """Empties the buffer and resets the position."""
        with self._lock:
            self.buffer.clear()
            self.position = 0

    def sample(
        self, batch_size: int, seed: Optional[int] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Samples a batch of transitions from the buffer.
        """
        return self.sample_query(batch_size, seed=seed)

    def sample_query(
        self,
        batch_size: int,
        filters: Optional[Dict[str, Any]] = None,
        temporal_window: Optional[int] = None,
        contiguous: bool = False,
        seed: Optional[int] = None,
    ) -> List[Any]:
        """
        Samples transitions from the buffer matching specific constraints.

        Args:
            batch_size: Number of items to sample.
            filters: Dictionary of metadata constraints (e.g., {'is_expert': True}).
            temporal_window: Only sample from the last N transitions.
            contiguous: If True, returns a single contiguous sequence of batch_size.
            seed: Random seed for sampling.
        """
        with self._lock:
            # Consume prefetch queue if available
            if hasattr(self, "_prefetch_queue") and self._prefetch_queue:
                return self._prefetch_queue.pop(0)

            if not self.buffer:
                return []

            # 1. Apply Temporal Window
            candidates = self.buffer
            if temporal_window:
                candidates = candidates[-temporal_window:]

            # 2. Apply Metadata Filters
            if filters:
                candidates = [
                    item for item in candidates if self._check_filters(item, filters)
                ]

            if not candidates:
                return []

            rng = random.Random(seed)

            # 3. Handle Contiguous Sampling
            if contiguous:
                if len(candidates) < batch_size:
                    return []
                start = rng.randint(0, len(candidates) - batch_size)
                return candidates[start : start + batch_size]

            # 4. Standard Random Sampling
            return rng.sample(candidates, min(len(candidates), batch_size))

    def _check_filters(self, item: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Deep check for filter matches in metadata."""
        for k, v in filters.items():
            # Support nested metadata check
            if k == "policy_version":
                # Special case for policy version in metadata/context
                try:
                    actor_id = list(
                        item["metadata"]["context"]["actor_snapshots"].keys()
                    )[0]
                    snap = item["metadata"]["context"]["actor_snapshots"][actor_id]
                    ver = snap["policy_version"] if isinstance(snap, dict) else snap
                    if ver != v:
                        return False
                except (KeyError, IndexError):
                    return False
            elif k == "on_policy":
                # Implementation: check if the data version matches the current policy version
                # For now, if provided as a boolean in metadata, just use the direct check
                if item.get("metadata", {}).get(k) != v:
                    return False
            else:
                # Direct metadata check
                if item.get("metadata", {}).get(k) != v:
                    return False
        return True

    def prefetch(
        self,
        batch_size: int,
        count: int = 1,
        filters: Optional[Dict[str, Any]] = None,
        temporal_window: Optional[int] = None,
        contiguous: bool = False,
    ):
        """Asynchronously prefetch samples matching a query in a background thread."""
        if not hasattr(self, "_prefetch_queue"):
            self._prefetch_queue = []

        def worker():
            for _ in range(count):
                sample = self.sample_query(
                    batch_size=batch_size,
                    filters=filters,
                    temporal_window=temporal_window,
                    contiguous=contiguous,
                )
                with self._lock:
                    self._prefetch_queue.append(sample)

        thread = threading.Thread(target=worker)
        thread.start()
        return thread

    def __len__(self) -> int:
        return len(self.buffer)


class ParameterStore:
    """
    Manages model state (parameters and buffers) and versioning.
    """

    def __init__(self, state: Dict[str, torch.Tensor]):
        self._state = state
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def get_state(self) -> Dict[str, torch.Tensor]:
        """Returns the current state (parameters and buffers)."""
        return self._state

    def update_state(self, new_state: Dict[str, torch.Tensor]) -> None:
        """
        Updates state in-place and increments version.
        Explicitly uses .copy_() to maintain reference integrity if needed.
        """
        with torch.no_grad():
            for name, tensor in new_state.items():
                if name in self._state:
                    self._state[name].copy_(tensor)
                else:
                    self._state[name] = tensor.detach().clone()
        self._version += 1


class OptimizerState:
    """
    Stores optimizer-specific state (e.g., moments, step count).
    Guarantees the full optimization lifecycle: zero_grad, backward, clip, and step.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, grad_clip: Optional[float] = None):
        self.optimizer = optimizer
        self.grad_clip = grad_clip

    def get_state(self) -> Dict[str, Any]:
        """Returns the internal state of the optimizer."""
        return self.optimizer.state_dict()

    def set_state(self, state_dict: Dict[str, Any]) -> None:
        """Sets the internal state of the optimizer."""
        self.optimizer.load_state_dict(state_dict)

    def step(self, loss: torch.Tensor) -> Dict[str, float]:
        """
        Performs a single optimization step.
        Guarantees: zero_grad, backward, grad_clipping, and step.
        
        Returns:
            Dictionary with grad_norm, lr, and loss.
        """
        self.optimizer.zero_grad(set_to_none=True)
        
        loss_value = loss.item()
        loss.backward()
        
        # Calculate grad norm and perform clipping
        grad_norm = 0.0
        params = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
        
        if params:
            # Use torch.nn.utils.clip_grad_norm_ to get norm even if clipping is disabled
            max_norm = self.grad_clip if self.grad_clip is not None else float('inf')
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm).item()

        self.optimizer.step()
        
        # Get current learning rate (from first param group)
        lr = self.optimizer.param_groups[0]['lr']
        
        return {
            "loss": loss_value,
            "grad_norm": grad_norm,
            "lr": lr
        }
