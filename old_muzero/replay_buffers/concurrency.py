import torch
import torch.multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union
from contextlib import contextmanager


class ConcurrencyBackend(ABC):
    """
    Abstract base class for concurrency backends in the replay buffer.
    Provides methods for creating tensors and locks that vary based on the
    underlying parallelization framework (Single Threaded, TorchMP, etc.).
    """

    @property
    @abstractmethod
    def is_shared(self) -> bool:
        """Returns True if the backend supports shared memory/multiprocessing."""
        pass  # pragma: no cover

    @abstractmethod
    def create_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        fill_value: Optional[Any] = None,
    ) -> torch.Tensor:
        """Creates a tensor appropriate for this backend."""
        pass  # pragma: no cover

    @abstractmethod
    def create_lock(self):
        """Creates a lock synchronization object appropriate for this backend."""
        pass  # pragma: no cover


class NoOpLock:
    """A context manager that does nothing, used for single-threaded backends."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # pragma: no cover

    def acquire(self):
        pass  # pragma: no cover

    def release(self):
        pass  # pragma: no cover


class LocalBackend(ConcurrencyBackend):
    """Standard backend for single-threaded or Ray-based (immutable objects) workflows."""

    @property
    def is_shared(self) -> bool:
        return False

    def create_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        fill_value: Optional[Any] = None,
    ) -> torch.Tensor:
        if fill_value is not None:
            return torch.full(shape, fill_value, dtype=dtype)
        return torch.zeros(shape, dtype=dtype)

    def create_lock(self):
        return NoOpLock()


class TorchMPBackend(ConcurrencyBackend):
    """Backend for Torch Multiprocessing using shared memory and mp.Lock."""

    @property
    def is_shared(self) -> bool:
        return True

    def create_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        fill_value: Optional[Any] = None,
    ) -> torch.Tensor:
        if fill_value is not None:
            tensor = torch.full(shape, fill_value, dtype=dtype)
        else:
            tensor = torch.zeros(shape, dtype=dtype)
        return tensor.share_memory_()

    def create_lock(self):
        return mp.Lock()
