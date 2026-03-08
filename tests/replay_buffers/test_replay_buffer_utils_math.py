import pytest
import numpy as np
from replay_buffers.utils import (
    sample_by_random_indices,
    sample_by_indices_probability,
    discounted_cumulative_sums,
)

pytestmark = pytest.mark.unit


def test_sample_by_random_indices():
    """Verifies standard and replacement sampling behavior."""
    np.random.seed(42)
    max_idx = 100

    # With replacement
    samples_repl = sample_by_random_indices(
        max_idx, batch_size=10, with_replacement=True
    )
    assert len(samples_repl) == 10
    assert all(0 <= x < 100 for x in samples_repl)

    # Without replacement
    samples_no_repl = sample_by_random_indices(
        max_idx, batch_size=10, with_replacement=False
    )
    assert len(set(samples_no_repl)) == 10  # All unique


def test_sample_by_indices_probability():
    """Verifies probability-weighted sampling selects highly weighted indices."""
    np.random.seed(42)
    probs = np.array([0.0, 0.0, 1.0, 0.0])  # 100% chance to pick index 2

    samples = sample_by_indices_probability(4, batch_size=5, probabilities=probs)
    assert np.array_equal(samples, np.array([2, 2, 2, 2, 2]))


def test_discounted_cumulative_sums():
    """Verifies correct mathematical computation of reward-to-go."""
    rewards = np.array([1.0, 1.0, 1.0])
    discount = 0.9

    # Expected: [1 + 0.9 + 0.81, 1 + 0.9, 1] => [2.71, 1.9, 1.0]
    expected = np.array([2.71, 1.9, 1.0])
    actual = discounted_cumulative_sums(rewards, discount)

    np.testing.assert_allclose(actual, expected, rtol=1e-5)
