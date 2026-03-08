import pytest
import torch
from search.search_py.root_policies import (
    VisitFrequencyPolicy,
    BestActionRootPolicy,
    CompletedQValuesRootPolicy,
)

pytestmark = pytest.mark.unit


class DummyConfig:
    gumbel_cvisit = 50.0
    gumbel_cscale = 1.0


class DummyMinMaxStats:
    def normalize(self, val):
        return val


class DummyChild:
    def __init__(self, expanded, val):
        self._expanded = expanded
        self._val = val

    def expanded(self):
        return self._expanded

    def value(self):
        return self._val


class DummyRootNode:
    def __init__(self):
        # 3 Actions
        self.child_visits = torch.tensor([10.0, 20.0, 0.0])
        self.child_values = torch.tensor([0.5, 0.8, 0.0])
        self.child_priors = torch.tensor([0.3, 0.6, 0.1])
        self.network_policy = torch.tensor([0.3, 0.6, 0.1])

        # Mocks for QValueScoring logic
        self.children = {
            0: DummyChild(True, 0.5),
            1: DummyChild(True, 0.8),
            2: DummyChild(False, 0.0),
        }

    def get_child_q_from_parent(self, child):
        return child.value()

    def get_child_q_for_unvisited(self):
        return 0.1

    def get_v_mix(self):
        return 0.2


def test_visit_frequency_policy():
    """Verifies proportional visit counts translate to standard AlphaZero policy."""
    policy_gen = VisitFrequencyPolicy(
        config=None, device=torch.device("cpu"), num_actions=3
    )
    root = DummyRootNode()

    policy = policy_gen.get_policy(root, DummyMinMaxStats())

    # [10, 20, 0] -> [1/3, 2/3, 0]
    assert torch.allclose(policy, torch.tensor([1 / 3, 2 / 3, 0.0]))


def test_best_action_root_policy():
    """Verifies greedy policy extraction targets the highest Q-value."""
    policy_gen = BestActionRootPolicy(
        config=None, device=torch.device("cpu"), num_actions=3
    )
    root = DummyRootNode()

    policy = policy_gen.get_policy(root, DummyMinMaxStats())

    # Action 1 has the highest value (0.8), so it should be a 1-hot vector
    assert torch.allclose(policy, torch.tensor([0.0, 1.0, 0.0]))


def test_completed_q_values_root_policy():
    """Verifies integration with the Gumbel Improved Policy utility."""
    torch.manual_seed(42)
    policy_gen = CompletedQValuesRootPolicy(
        config=DummyConfig(), device=torch.device("cpu"), num_actions=3
    )
    root = DummyRootNode()

    policy = policy_gen.get_policy(root, DummyMinMaxStats())

    # Must be a valid probability distribution
    assert torch.allclose(policy.sum(), torch.tensor(1.0))
    assert policy.shape == (3,)
    assert (policy >= 0).all()
