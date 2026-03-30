import numpy as np
from abc import ABC, abstractmethod

import torch


class Writer(ABC):
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.size = 0

    @abstractmethod
    def store(self) -> int | None:
        """
        Determines the index to write to.
        Returns the index, or None if the data should be discarded.
        Updates internal size/pointer state.
        """
        pass  # pragma: no cover

    @abstractmethod
    def store_batch(self, batch_size: int) -> list[slice] | None:
        """
        Reserves a block of indices for batch writing.
        Returns a list of slices (e.g., [slice(0, 10)] or [slice(90, 100), slice(0, 5)] for wrap-around).
        Returns None if the batch cannot be added.
        """
        pass  # pragma: no cover

    def clear(self):
        self.size = 0


class CircularWriter(Writer):
    """
    Standard FIFO circular buffer. Overwrites oldest data when full.
    """

    def __init__(self, max_size: int):
        super().__init__(max_size)
        self.pointer = 0

    def store(self) -> int | None:
        idx = self.pointer
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        return idx

    def store_batch(self, batch_size: int) -> list[slice] | None:
        start = self.pointer
        end = start + batch_size

        slices = []
        if end <= self.max_size:
            slices.append(slice(start, end))
            self.pointer = (self.pointer + batch_size) % self.max_size
        else:
            # Wrap around
            first_part = self.max_size - start
            remainder = batch_size - first_part
            slices.append(slice(start, self.max_size))
            slices.append(slice(0, remainder))
            self.pointer = remainder

        self.size = min(self.size + batch_size, self.max_size)
        return slices

    def clear(self):
        super().clear()
        self.pointer = 0


class SharedCircularWriter(CircularWriter):
    """
    Circular Writer that uses PyTorch Shared Memory tensors for state.
    Essential for Multiprocessing (e.g. MuZero).
    """

    def __init__(self, max_size: int):
        # We don't call super().__init__ because we manage size/pointer differently
        self.max_size = max_size
        self._pointer = torch.zeros(1, dtype=torch.int64).share_memory_()
        self._size = torch.zeros(1, dtype=torch.int64).share_memory_()

    @property
    def pointer(self):
        return int(self._pointer.item())

    @pointer.setter
    def pointer(self, value):
        self._pointer[0] = value

    @property
    def size(self):
        return int(self._size.item())

    @size.setter
    def size(self, value):
        self._size[0] = value

    def clear(self):
        self.pointer = 0
        self.size = 0


class ReservoirWriter(Writer):
    """
    Reservoir sampling buffer.
    When full, new data replaces existing data with probability (max_size / total_seen).
    """

    def __init__(self, max_size: int):
        super().__init__(max_size)
        self.add_calls = 0

    def store(self) -> int | None:
        idx = None
        if self.size < self.max_size:
            idx = self.size
            self.size += 1
        else:
            # Reservoir logic: replace existing item with prob max_size / k
            # Equivalent to picking random int in [0, k] and keeping if < max_size
            r = np.random.randint(0, self.add_calls + 1)
            if r < self.max_size:
                idx = r

        self.add_calls += 1
        return idx

    def store_batch(self, batch_size):
        raise NotImplementedError(
            "Batch store not implemented for ReservoirWriter."
        )  # pragma: no cover

    def clear(self):
        super().clear()
        self.add_calls = 0


class PPOWriter(Writer):
    """
    Sequential writer for On-Policy algorithms (like PPO).
    Raises an error if an attempt is made to write beyond max_size.
    """

    def __init__(self, max_size: int):
        super().__init__(max_size)
        self.pointer = 0

    def store(self) -> int | None:
        if self.size >= self.max_size:
            raise IndexError(
                f"PPO Buffer Overflow: Attempted to write beyond max_size ({self.max_size}). PPO buffer should be cleared after sampling."
            )

        idx = self.pointer
        self.pointer += 1
        self.size += 1
        return idx

    def store_batch(self, batch_size: int) -> list[slice] | None:
        if self.size + batch_size > self.max_size:
            raise IndexError(
                f"PPO Buffer Overflow: Attempted to write {batch_size} beyond max_size ({self.max_size}). PPO buffer should be cleared after sampling."
            )

        start = self.pointer
        end = start + batch_size
        self.pointer += batch_size
        self.size += batch_size
        return [slice(start, end)]

    def clear(self):
        super().clear()
        self.pointer = 0
