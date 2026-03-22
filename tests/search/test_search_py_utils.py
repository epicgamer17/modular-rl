import pytest
import torch
from search.search_py.utils import get_completed_q, get_completed_q_improved_policy
from tests.search.conftest import DummyMinMaxStats, DummySearchConfig, make_dummy_utils_node

pytestmark = pytest.mark.unit


def test_get_completed_q_visited_and_bootstrap_logic():
    """Verifies unvisited actions get bootstrapped and visited actions use child values."""
    node = make_dummy_utils_node()
    stats = DummyMinMaxStats(normalize_fn=lambda val: val / 10.0)

    q_vals = get_completed_q(node, stats)

    # Action 0 is Unvisited: normalized bootstrap (-1.0 / 10.0) = -0.1
    # Action 1 is Visited: normalized child value (5.0 / 10.0) = 0.5
    # Action 2 is Visited: normalized child value (2.0 / 10.0) = 0.2
    assert torch.allclose(q_vals, torch.tensor([-0.1, 0.5, 0.2]))


def test_get_completed_q_improved_policy_distribution():
    """Verifies the resulting policy pi0 is a valid, normalized probability distribution."""
    torch.manual_seed(42)
    node = make_dummy_utils_node()
    stats = DummyMinMaxStats(normalize_fn=lambda val: val / 10.0)
    config = DummySearchConfig()

    pi0 = get_completed_q_improved_policy(config, node, stats)

    # Must sum to 1.0 (valid probability space)
    assert torch.allclose(pi0.sum(), torch.tensor(1.0))
    assert pi0.shape == (3,)
    # Cannot contain negative probabilities
    assert (pi0 >= 0).all()
