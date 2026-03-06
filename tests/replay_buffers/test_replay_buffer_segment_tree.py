import pytest
pytestmark = pytest.mark.unit

import torch
from replay_buffers.segment_tree import SumSegmentTree, MinSegmentTree
from replay_buffers.concurrency import LocalBackend, TorchMPBackend


def test_sum_segment_tree():
    capacity = 8
    backend = LocalBackend()
    tree = SumSegmentTree(capacity, backend=backend)

    # Test basic setting and getting
    tree[0] = 1.0
    tree[1] = 2.0
    assert tree[0] == 1.0
    assert tree[1] == 2.0

    # Test sum operation
    assert tree.sum(0, 1) == 3.0
    assert tree.sum(0, 7) == 3.0

    tree[2] = 3.0
    assert tree.sum(0, 2) == 6.0
    assert tree.sum(1, 2) == 5.0

    # Test retrieve
    # tree values: [1, 2, 3, 0, 0, 0, 0, 0]
    # cumulative: [1, 3, 6, 6, 6, 6, 6, 6]
    assert tree.retrieve(0.5) == 0
    assert tree.retrieve(1.5) == 1
    assert tree.retrieve(3.5) == 2
    assert tree.retrieve(5.9) == 2


def test_min_segment_tree():
    capacity = 8
    backend = LocalBackend()
    tree = MinSegmentTree(capacity, backend=backend)

    tree[0] = 10.0
    tree[1] = 5.0
    tree[2] = 20.0

    assert tree.min(0, 2) == 5.0
    assert tree.min(0, 0) == 10.0
    assert tree.min(2, 2) == 20.0

    assert tree.min_index(0, 2) == 1 + capacity  # Index in tree storage


def test_torch_mp_backend():
    capacity = 4
    backend = TorchMPBackend()
    try:
        tree = SumSegmentTree(capacity, backend=backend)
    except RuntimeError as err:
        if "Operation not permitted" in str(err) or "torch_shm_manager" in str(err):
            pytest.skip("Shared-memory backend is unavailable in this test environment")
        raise

    assert tree.tree.is_shared()
    tree[0] = 1.0
    assert tree.sum(0, 0) == 1.0
