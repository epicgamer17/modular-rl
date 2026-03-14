import pytest
pytest.importorskip("hypothesis")
pytestmark = pytest.mark.unit

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

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


@settings(max_examples=60, deadline=None)
@given(capacity_exp=st.integers(min_value=1, max_value=8), data=st.data())
def test_sum_segment_tree_leaf_sum_equals_root_property(capacity_exp, data):
    capacity = 2**capacity_exp
    leaves = data.draw(
        st.lists(
            st.floats(
                min_value=0.0,
                max_value=1_000.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            min_size=capacity,
            max_size=capacity,
        )
    )

    tree = SumSegmentTree(capacity, backend=LocalBackend())
    for idx, val in enumerate(leaves):
        tree[idx] = val

    expected_sum = torch.tensor(leaves, dtype=torch.float32).sum()
    root_sum = tree.tree[1]
    range_sum = tree.sum(0, capacity - 1)

    assert torch.allclose(root_sum, expected_sum, atol=1e-4, rtol=1e-5)
    assert torch.allclose(range_sum, expected_sum, atol=1e-4, rtol=1e-5)


@settings(max_examples=60, deadline=None)
@given(capacity_exp=st.integers(min_value=1, max_value=8), data=st.data())
def test_min_segment_tree_root_matches_leaf_min_property(capacity_exp, data):
    capacity = 2**capacity_exp
    leaves = data.draw(
        st.lists(
            st.floats(
                min_value=-1_000.0,
                max_value=1_000.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            min_size=capacity,
            max_size=capacity,
        )
    )

    tree = MinSegmentTree(capacity, backend=LocalBackend())
    for idx, val in enumerate(leaves):
        tree[idx] = val

    expected_min = torch.tensor(leaves, dtype=torch.float32).min()
    root_min = tree.min(0, capacity - 1)
    min_leaf_index = tree.min_index(0, capacity - 1) - capacity

    assert torch.allclose(root_min, expected_min, atol=1e-4, rtol=1e-5)
    assert abs(float(tree[min_leaf_index]) - float(expected_min)) <= 1e-4
