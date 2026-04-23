"""
Stateful components for the RL IR runtime.
Includes ReplayBuffer, ParameterStore, and OptimizerState.
Follows the rule of explicit mutation and no hidden global state.
"""

from typing import List, Dict, Any, Optional
import torch
import random
import threading

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

    def sample(self, batch_size: int, seed: Optional[int] = None) -> List[Dict[str, torch.Tensor]]:
        """
        Samples a batch of transitions from the buffer.
        """
        if hasattr(self, "_prefetch_queue") and self._prefetch_queue:
            return self._prefetch_queue.pop(0)

        with self._lock:
            if not self.buffer:
                return []
                
            rng = random.Random(seed)
            return rng.sample(self.buffer, min(len(self.buffer), batch_size))

    def prefetch(self, batch_size: int, count: int = 1):
        """Asynchronously prefetch samples in a background thread."""
        if not hasattr(self, "_prefetch_queue"):
            self._prefetch_queue = []

        def worker():
            for _ in range(count):
                sample = self.sample(batch_size)
                self._prefetch_queue.append(sample)

        thread = threading.Thread(target=worker)
        thread.start()
        return thread

    def __len__(self) -> int:
        return len(self.buffer)

class ParameterStore:
    """
    Manages model parameters and versioning.
    """
    def __init__(self, parameters: Dict[str, torch.Tensor]):
        self._params = parameters
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Returns the current parameters."""
        return self._params

    def update_parameters(self, new_params: Dict[str, torch.Tensor]) -> None:
        """
        Updates parameters in-place and increments version.
        Explicitly uses .copy_() to maintain reference integrity if needed.
        """
        with torch.no_grad():
            for name, param in new_params.items():
                if name in self._params:
                    self._params[name].copy_(param)
                else:
                    self._params[name] = param.detach().clone()
        self._version += 1

class OptimizerState:
    """
    Stores optimizer-specific state (e.g., moments, step count).
    """
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def get_state(self) -> Dict[str, Any]:
        """Returns the internal state of the optimizer."""
        return self.optimizer.state_dict()

    def set_state(self, state_dict: Dict[str, Any]) -> None:
        """Sets the internal state of the optimizer."""
        self.optimizer.load_state_dict(state_dict)

    def step(self, loss: torch.Tensor) -> None:
        """Performs a single optimization step."""
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
