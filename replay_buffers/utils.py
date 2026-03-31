import numpy as np
import numpy.typing as npt
import scipy
from dataclasses import dataclass
import torch


def sample_by_random_indices(
    max_index_or_1darray, batch_size: int, with_replacement=False
) -> npt.NDArray[np.int64]:
    """
    Sample from a numpy array using indices
    """
    return np.random.choice(max_index_or_1darray, batch_size, replace=with_replacement)


def sample_by_indices_probability(
    max_index_or_1darray, batch_size: int, probabilities: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    """
    Sample from a numpy array using indices
    """
    return np.random.choice(max_index_or_1darray, batch_size, p=probabilities)


def sample_tree_proportional(
    tree, batch_size: int, max_size: int
) -> npt.NDArray[np.int64]:
    """
    tree: SumSegmentTree
    Sample proportionally from a sum segment tree. Used in prioritized experience replay
    """
    indices = np.zeros(batch_size, dtype=np.int64)
    total_priority = tree.sum(0, max_size - 1)
    priority_segment = total_priority / batch_size

    for i in range(batch_size):
        l = priority_segment * i
        h = priority_segment * (i + 1)
        upperbound = np.random.uniform(l, h)
        indices[i] = tree.retrieve(upperbound)
        # print(tree[indices[i]])

    return indices


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
