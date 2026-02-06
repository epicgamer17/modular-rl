import numpy as np
import torch
from abc import ABC, abstractmethod
from replay_buffers.segment_tree import SumSegmentTree, MinSegmentTree


class Sampler(ABC):
    @abstractmethod
    def sample(self, buffer_size: int, batch_size: int):
        """Returns indices and optionally weights."""
        pass

    def on_store(self, idx: int, priority: float = None, **kwargs):
        """Hook called when data is stored in the buffer."""
        pass

    def update_priorities(self, indices, priorities, **kwargs):
        pass

    def clear(self):
        pass

    def set_beta(self, beta):
        pass


class UniformSampler(Sampler):
    def sample(self, buffer_size: int, batch_size: int):
        indices = np.random.choice(buffer_size, batch_size, replace=False)
        return indices, None


class WholeBufferSampler(Sampler):
    def sample(self, buffer_size: int, batch_size: int = None):
        # Return all indices
        return np.arange(buffer_size), None


class PrioritizedSampler(Sampler):
    def __init__(
        self,
        max_size: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
        max_priority: float = 1.0,
        use_batch_weights: bool = False,
        use_initial_max_priority: bool = True,
        is_shared: bool = True,
    ):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.max_priority = max_priority
        self.use_initial_max_priority = use_initial_max_priority
        self.use_batch_weights = use_batch_weights
        self.is_shared = is_shared

        self.initial_max_priority = max_priority

        tree_capacity = 1
        while tree_capacity < max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity, is_shared=is_shared)
        self.min_tree = MinSegmentTree(tree_capacity, is_shared=is_shared)

    def sample(self, buffer_size: int, batch_size: int):
        indices = self._sample_proportional(buffer_size, batch_size)

        assert all(
            idx < buffer_size for idx in indices
        ), f"Sampled index exceeds current buffer size, indices: {indices}, sum_tree: {self.sum_tree}, min_tree: {self.min_tree}"

        weights = torch.tensor(
            [self._calculate_weight(i, buffer_size) for i in indices],
            dtype=torch.float32,
        )

        if self.use_batch_weights:
            weights = weights / weights.max()
        else:
            # Importance sampling weights normalization
            min_priority = self.min_tree.min() / self.sum_tree.sum()
            # Avoid divide by zero if tree is empty or min_priority is 0 (though min() init is inf)
            if min_priority == 0:
                min_priority = 1e-10

            max_weight = (min_priority * buffer_size) ** (-self.beta)
            weights = weights / max_weight

        return indices, weights

    def _sample_proportional(self, buffer_size, batch_size):
        indices = []
        total_priority = self.sum_tree.sum(0, buffer_size - 1)

        if total_priority == 0:
            raise ValueError(
                f"Total priority is zero, cannot sample. buffer_size: {buffer_size}, batch_size: {batch_size}"
            )

        priority_segment = total_priority / batch_size
        # print(f"priority_segment: {priority_segment}")
        # print(f"total_priority: {total_priority}")
        assert priority_segment * batch_size == total_priority
        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            # Ensure upperbound is within [0, total_priority] and slightly less than total_priority
            # to avoid retrieve returning buffer_size if it hits the boundary precisely.
            upperbound = min(upperbound, total_priority - 1e-6)
            upperbound = max(0.0, upperbound)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def _calculate_weight(self, index: int, buffer_size: int):
        priority_sample = self.sum_tree[index] / self.sum_tree.sum()
        weight = (priority_sample * buffer_size) ** (-self.beta)
        return weight

    def on_store(
        self, idx: int, priority: float = None, sum_tree_val=None, min_tree_val=None
    ):
        """Updates the tree at index idx with specific priority or raw tree values."""
        if priority is None:
            priority = self.max_priority

        self.sum_tree[idx] = (
            sum_tree_val if sum_tree_val is not None else priority**self.alpha
        )
        self.min_tree[idx] = (
            min_tree_val if min_tree_val is not None else priority**self.alpha
        )
        self.max_priority = max(self.max_priority, priority)

    def update_priorities(self, indices, priorities, ids=None, buffer_ids=None):
        priorities = priorities + self.epsilon
        if ids is not None:
            assert (
                len(priorities) == len(ids) == len(indices)
                or priorities.shape == ids.shape == indices.shape
            )

            for index, id, priority in zip(indices, ids, priorities):
                assert (
                    priority > 0
                ), "Negative priority: {} \n All priorities {}".format(
                    priority, priorities
                )
                # TODO: ADD THIS ASSERT BACK SOME HOW
                # assert 0 <= index < len(self)

                if buffer_ids[index] != id:
                    continue

                self.sum_tree[index] = priority**self.alpha
                self.min_tree[index] = priority**self.alpha
                self.max_priority = max(self.max_priority, priority)
        else:
            assert len(indices) == len(priorities)
            for index, priority in zip(indices, priorities):
                assert priority > 0, "Negative priority: {}".format(priority)
                # TODO: ADD THIS ASSERT BACK SOME HOW
                # assert 0 <= index < len(self)

                self.sum_tree[index] = priority**self.alpha
                self.min_tree[index] = priority**self.alpha
                self.max_priority = max(
                    self.max_priority, priority
                )  # could remove and clip priorities in experience replay isntead

        return priorities**self.alpha

    def set_beta(self, beta):
        self.beta = beta

    def clear(self):
        # Re-initialize trees or zero them out
        capacity = self.sum_tree.capacity
        self.sum_tree = SumSegmentTree(capacity, is_shared=self.is_shared)
        self.min_tree = MinSegmentTree(capacity, is_shared=self.is_shared)
        self.max_priority = 1.0
