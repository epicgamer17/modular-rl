"""Data system: replay buffers."""

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

    def add(self, transition: Any) -> None:
        """Adds a transition to the buffer."""
        if not isinstance(transition, dict):
            # Convert TransitionBatch or other dataclasses to dict
            if hasattr(transition, "__dataclass_fields__"):
                transition = {
                    f: getattr(transition, f) for f in transition.__dataclass_fields__
                }
            else:
                raise TypeError(f"Expected dict or dataclass, got {type(transition)}")

        new_entry = {
            k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in transition.items()
            if v is not None # Don't store None fields
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
