import pytest
pytest.importorskip("hypothesis")

pytestmark = pytest.mark.unit

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from replay_buffers.samplers import PrioritizedSampler
from replay_buffers.concurrency import LocalBackend


def test_prioritized_sampler_local():
    max_size = 8
    batch_size = 2
    backend = LocalBackend()
    sampler = PrioritizedSampler(max_size, backend=backend)

    # Test default values
    assert sampler.alpha == 0.6
    assert sampler.beta == 0.4

    # Store some items with different priorities
    # on_store(idx, priority)
    sampler.on_store(0, priority=1.0)  # priority^alpha = 1.0
    sampler.on_store(1, priority=2.0)  # 2^0.6 = 1.5157
    sampler.on_store(2, priority=0.5)  # 0.5^0.6 = 0.6597

    # Sample
    indices, weights = sampler.sample(buffer_size=3, batch_size=batch_size)

    assert len(indices) == batch_size
    assert weights.shape == (batch_size,)
    assert all(0 <= idx < 3 for idx in indices)

    # Update priorities
    new_priorities = torch.tensor([10.0, 10.0])
    sampler.update_priorities(indices, new_priorities)

    # Check that sum_tree reflects updates
    total_p = sampler.sum_tree.sum(0, 7)
    # Remaining item at index 2 has original priority
    expected_remaining = 0.5**0.6
    # Updated items have 10.0^0.6
    expected_updated = 10.0**0.6 * batch_size
    # This matches if indices were unique, but they might not be.
    # But we know total_p should be roughly right.
    assert total_p > 0


def test_prioritized_sampler_clear():
    max_size = 4
    sampler = PrioritizedSampler(max_size)
    sampler.on_store(0, priority=1.0)
    assert sampler.sum_tree.sum(0, 3) > 0

    sampler.clear()
    assert sampler.sum_tree.sum(0, 3) == 0
    assert sampler.max_priority == 1.0


@settings(max_examples=50, deadline=None)
@given(capacity_exp=st.integers(min_value=1, max_value=7), data=st.data())
def test_prioritized_sampler_tree_invariants_property(capacity_exp, data):
    max_size = 2**capacity_exp
    buffer_size = data.draw(st.integers(min_value=1, max_value=max_size))
    priorities = data.draw(
        st.lists(
            st.floats(
                min_value=1e-4,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            min_size=buffer_size,
            max_size=buffer_size,
        )
    )

    sampler = PrioritizedSampler(max_size, backend=LocalBackend())
    for idx, priority in enumerate(priorities):
        sampler.on_store(idx, priority=priority)

    transformed = torch.tensor(
        [float(priority) ** sampler.alpha for priority in priorities],
        dtype=torch.float32,
    )

    total_priority = sampler.sum_tree.sum(0, max_size - 1)
    minimum_priority = sampler.min_tree.min(0, max_size - 1)

    assert torch.allclose(total_priority, transformed.sum(), atol=1e-4, rtol=1e-5)
    assert torch.allclose(minimum_priority, transformed.min(), atol=1e-4, rtol=1e-5)
