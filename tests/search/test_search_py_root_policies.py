import pytest
import torch
from search.search_py.root_policies import (
    VisitFrequencyPolicy,
    BestActionRootPolicy,
    CompletedQValuesRootPolicy,
)
from tests.search.conftest import (
    DummyMinMaxStats,
    DummySearchConfig,
    make_dummy_root_node,
)

pytestmark = pytest.mark.unit


def test_visit_frequency_policy():
    """Verifies proportional visit counts translate to standard AlphaZero policy."""
    policy_gen = VisitFrequencyPolicy(
        config=None, device=torch.device("cpu"), num_actions=3
    )
    root = make_dummy_root_node()

    policy = policy_gen.get_policy(root, DummyMinMaxStats(normalize=True))

    # [10, 20, 0] -> [1/3, 2/3, 0]
    assert torch.allclose(policy, torch.tensor([1 / 3, 2 / 3, 0.0]))


def test_best_action_root_policy():
    """Verifies greedy policy extraction targets the highest Q-value."""
    policy_gen = BestActionRootPolicy(
        config=None, device=torch.device("cpu"), num_actions=3
    )
    root = make_dummy_root_node()

    policy = policy_gen.get_policy(root, DummyMinMaxStats(normalize=True))

    # Action 1 has the highest value (0.8), so it should be a 1-hot vector
    assert torch.allclose(policy, torch.tensor([0.0, 1.0, 0.0]))


def test_completed_q_values_root_policy():
    """Verifies integration with the Gumbel Improved Policy utility."""
    torch.manual_seed(42)
    policy_gen = CompletedQValuesRootPolicy(
        config=DummySearchConfig(), device=torch.device("cpu"), num_actions=3
    )
    root = make_dummy_root_node()

    policy = policy_gen.get_policy(root, DummyMinMaxStats(normalize=True))

    # Must be a valid probability distribution
    assert torch.allclose(policy.sum(), torch.tensor(1.0))
    assert policy.shape == (3,)
    assert (policy >= 0).all()
