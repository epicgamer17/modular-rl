from typing import Callable, Optional

import torch

from data.concurrency import ConcurrencyBackend, LocalBackend


class SegmentTree:
    """Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    Attributes:
        capacity (int)
        tree (list)
        operation (function)

    """

    def __init__(
        self,
        capacity: int,
        operation: Callable,
        init_value: float,
        backend: Optional[ConcurrencyBackend] = None,
    ):
        """Initialization.

        Args:
            capacity (int)
            operation (function)
            init_value (float)
            backend (ConcurrencyBackend)

        """
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.backend = backend or LocalBackend()
        self.tree = self.backend.create_tensor(
            (2 * capacity,), dtype=torch.float32, fill_value=init_value
        )
        self.operation = operation

    def _operate_helper(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def _get_operate_index_helper(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> int:
        """Returns index of result of operation in segment."""
        if start == node_start and end == node_end:
            return node
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._get_operate_index_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._get_operate_index_helper(
                    start, end, 2 * node + 1, mid + 1, node_end
                )
            else:
                left_index = self._get_operate_index_helper(
                    start, mid, 2 * node, node_start, mid
                )
                right_index = self._get_operate_index_helper(
                    mid + 1, end, 2 * node + 1, mid + 1, node_end
                )
                if (
                    self.tree[left_index].item()
                    == self.operation(
                        self.tree[left_index], self.tree[right_index]
                    ).item()
                ):
                    return left_index
                else:
                    return right_index

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`.

        Args:
            start (int): Start index (inclusive).
            end (int): End index (inclusive). If None, uses capacity - 1.
        """
        # if end is None:
        #     end = self.capacity - 1
        # elif end < 0:
        #     end += self.capacity

        assert (
            0 <= start <= end < self.capacity
        ), f"Invalid range: [{start}, {end}] for capacity {self.capacity}"

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def get_operate_index(self, start: int = 0, end: int = 0) -> int:
        """Returns index of the result of applying `self.operation`.

        Args:
            start (int): Start index (inclusive).
            end (int): End index (inclusive). If None, uses capacity - 1.
        """
        # if end is None:
        #     end = self.capacity - 1
        # elif end < 0:
        #     end += self.capacity

        assert (
            0 <= start <= end < self.capacity
        ), f"Invalid range: [{start}, {end}] for capacity {self.capacity}"

        return self._get_operate_index_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        # FIX: cast to float to avoid TypeError when assigning numpy types to torch tensors
        idx += self.capacity
        self.tree[idx] = float(val)

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """Create SumSegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(
        self,
        capacity: int,
        backend: Optional[ConcurrencyBackend] = None,
    ):
        """Initialization.

        Args:
            capacity (int)
            backend (ConcurrencyBackend)

        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=torch.add,
            init_value=0.0,
            backend=backend,
        )

    def sum(self, start: int = 0, end: Optional[int] = None) -> float:
        """Returns arr[start] + ... + arr[end]."""
        if end is None:
            end = self.capacity - 1
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        # TODO: Check assert case and fix bug
        assert (
            0 <= upperbound <= self.sum(0, self.capacity - 1) + 1e-5
        ), f"upperbound: {upperbound} < {self.sum(0, self.capacity - 1) + 1e-5}"

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left].item() > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left].item()
                idx = right
        assert (idx - self.capacity) <= self.capacity
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(
        self,
        capacity: int,
        backend: Optional[ConcurrencyBackend] = None,
    ):
        """Initialization.

        Args:
            capacity (int)
            backend (ConcurrencyBackend)

        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=torch.minimum,
            init_value=float("inf"),
            backend=backend,
        )

    def min(self, start: int = 0, end: Optional[int] = None) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        if end is None:
            end = self.capacity - 1
        return super(MinSegmentTree, self).operate(start, end)

    def min_index(self, start: int = 0, end: int = 0) -> int:
        """Returns index of min(arr[start], ...,  arr[end])."""
        node = super(MinSegmentTree, self).get_operate_index(start, end)
        while node < self.capacity:
            left = 2 * node
            right = left + 1
            if self.tree[left].item() <= self.tree[right].item():
                node = left
            else:
                node = right
        return node
