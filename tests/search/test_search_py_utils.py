import pytest
import torch
from search.search_py.utils import get_completed_q, get_completed_q_improved_policy

pytestmark = pytest.mark.unit


class DummyMinMaxStats:
    def normalize(self, val):
        # Identity shift for simple math testing
        return val / 10.0


class DummyNode:
    def __init__(self):
        # Action 0: Unvisited, Action 1: Visited, Action 2: Visited
        self.child_priors = torch.tensor([0.1, 0.7, 0.2])
        self.network_policy = torch.tensor([0.2, 0.6, 0.2])
        self.child_visits = torch.tensor([0, 5, 2])
        self.child_values = torch.tensor([0.0, 5.0, 2.0])

    def get_v_mix(self):
        return torch.tensor(1.0)

    def get_child_q_for_unvisited(self):
        return torch.tensor(-1.0)


class DummySearchConfig:
    gumbel_cvisit = 50.0
    gumbel_cscale = 1.0


def test_get_completed_q_visited_and_bootstrap_logic():
    """Verifies unvisited actions get bootstrapped and visited actions use child values."""
    node = DummyNode()
    stats = DummyMinMaxStats()

    q_vals = get_completed_q(node, stats)

    # Action 0 is Unvisited: normalized bootstrap (-1.0 / 10.0) = -0.1
    # Action 1 is Visited: normalized child value (5.0 / 10.0) = 0.5
    # Action 2 is Visited: normalized child value (2.0 / 10.0) = 0.2
    assert torch.allclose(q_vals, torch.tensor([-0.1, 0.5, 0.2]))


def test_get_completed_q_improved_policy_distribution():
    """Verifies the resulting policy pi0 is a valid, normalized probability distribution."""
    torch.manual_seed(42)
    node = DummyNode()
    stats = DummyMinMaxStats()
    config = DummySearchConfig()

    pi0 = get_completed_q_improved_policy(config, node, stats)

    # Must sum to 1.0 (valid probability space)
    assert torch.allclose(pi0.sum(), torch.tensor(1.0))
    assert pi0.shape == (3,)
    # Cannot contain negative probabilities
    assert (pi0 >= 0).all()
