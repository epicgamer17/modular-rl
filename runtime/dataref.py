"""
DataRef abstraction for the RL IR.
Provides a unified interface for referencing data (tensors, buffers, streams)
with explicit location awareness, lifetime management, and transfer tracking.
"""

from enum import Enum
from typing import Any, Optional, List, Dict
import torch
import time
import uuid


class StorageLocation(Enum):
    CPU = "cpu"
    GPU = "gpu"
    SHARED_MEMORY = "shared_memory"
    REMOTE = "remote"


class DataRef:
    """
    A location-aware reference to a piece of data.
    Ensures that the IR can track where data resides and its movement history.

    Attributes:
        data: The underlying data (initially a PyTorch tensor).
        location: The physical location of the data (CPU, GPU, SHARED_MEMORY, REMOTE).
        ref_id: A unique identifier for this data reference.
        created_at: The timestamp when this data reference was created.
        lifetime_metadata: Metadata associated with the lifetime of this data reference.
        transfer_history: A history of data transfers.
    """

    def __init__(
        self,
        data: Any,
        location: StorageLocation = StorageLocation.CPU,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._data = data
        self.location = location
        self.ref_id = str(uuid.uuid4())
        self.created_at = time.time()
        self.lifetime_metadata = metadata or {}
        self.transfer_history: List[Dict[str, Any]] = [
            {"timestamp": self.created_at, "to": location.value, "reason": "creation"}
        ]

    @property
    def data(self) -> Any:
        return self._data

    def move_to(self, new_location: StorageLocation, device_id: Optional[int] = None):
        """
        Moves the underlying data to a new physical location.
        Records the transfer in the history.
        """
        if new_location == self.location:
            return

        old_location = self.location

        # Physical transfer logic (specialized for PyTorch tensors)
        if isinstance(self._data, torch.Tensor):
            if new_location == StorageLocation.GPU:
                if torch.cuda.is_available():
                    target_device = f"cuda:{device_id}" if device_id is not None else "cuda"
                elif torch.backends.mps.is_available():
                    target_device = "mps"
                else:
                    target_device = "cpu" # Fallback to CPU if no GPU available
                self._data = self._data.to(target_device)
            elif new_location == StorageLocation.CPU:
                self._data = self._data.to("cpu")
            # SHARED_MEMORY and REMOTE would require more complex logic (multiprocessing.shared_memory, etc.)

        self.location = new_location
        self.transfer_history.append(
            {
                "timestamp": time.time(),
                "from": old_location.value,
                "to": new_location.value,
                "reason": "explicit_move",
            }
        )

    def __repr__(self) -> str:
        shape = getattr(self._data, "shape", "N/A")
        return (
            f"{self.__class__.__name__}(id={self.ref_id[:8]}, "
            f"loc={self.location.value}, shape={shape})"
        )


class BufferRef(DataRef):
    """A reference to a persistent buffer (e.g., replay buffer slice)."""

    pass


class StreamRef(DataRef):
    """A reference to a stream of data (e.g., environment observations)."""

    pass
