from dataclasses import dataclass
from typing import Any, Optional, Dict, List
from enum import Enum
import uuid
import time
import torch


from runtime.base import RuntimeValue


@dataclass(frozen=True)
class Value(RuntimeValue):
    """A standard data payload."""

    data: Any

    @property
    def has_data(self) -> bool:
        return True

    def __repr__(self):
        return f"Value({self.data})"

    def __bool__(self):
        return True


class StorageLocation(Enum):
    CPU = "cpu"
    GPU = "gpu"
    SHARED_MEMORY = "shared_memory"
    REMOTE = "remote"


class DataRef(RuntimeValue):
    """
    A location-aware reference to a piece of data.
    Ensures that the IR can track where data resides and its movement history.
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

    @property
    def has_data(self) -> bool:
        return True

    def move_to(self, new_location: StorageLocation, device_id: Optional[int] = None):
        """
        Moves the underlying data to a new physical location.
        """
        if new_location == self.location:
            return

        old_location = self.location
        from core.batch import TransitionBatch

        def move_tensor(t, loc, dev_id):
            if not isinstance(t, torch.Tensor):
                return t
            if loc == StorageLocation.GPU:
                if torch.cuda.is_available():
                    target_device = f"cuda:{dev_id}" if dev_id is not None else "cuda"
                elif torch.backends.mps.is_available():
                    target_device = "mps"
                else:
                    target_device = "cpu"
                return t.to(target_device)
            elif loc == StorageLocation.CPU:
                return t.to("cpu")
            return t

        if isinstance(self._data, torch.Tensor):
            self._data = move_tensor(self._data, new_location, device_id)
        elif isinstance(self._data, TransitionBatch):
            from dataclasses import replace

            updates = {}
            for field in self._data.__dataclass_fields__:
                val = getattr(self._data, field)
                if isinstance(val, torch.Tensor):
                    updates[field] = move_tensor(val, new_location, device_id)
            self._data = replace(self._data, **updates)

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
            f"DataRef(id={self.ref_id[:8]}, "
            f"loc={self.location.value}, shape={shape})"
        )

    def __bool__(self):
        return True


class BufferRef(DataRef):
    """A reference to a persistent buffer (e.g., replay buffer slice)."""

    pass


class StreamRef(DataRef):
    """A reference to a stream of data (e.g., environment observations)."""

    pass
