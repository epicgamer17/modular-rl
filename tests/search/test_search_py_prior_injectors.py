import pytest
import numpy as np
import torch
from types import SimpleNamespace
from search.search_py.prior_injectors import (
    DirichletInjector,
    ActionTargetInjector,
    GumbelInjector,
    _safe_log_probs,
)

pytestmark = pytest.mark.unit


def test_safe_log_probs():
    """Verifies zeros map to exactly -inf without NaN issues."""
    probs = torch.tensor([0.5, 0.0, 1.0])
    logits = _safe_log_probs(probs)
    assert logits[1].item() == -float("inf")
    assert torch.isclose(logits[2], torch.tensor(0.0))


def test_dirichlet_injector():
    """Verifies Dirichlet noise is properly weighted into the legal moves."""
    np.random.seed(42)
    torch.manual_seed(42)
    config = SimpleNamespace(
        use_dirichlet=True, dirichlet_alpha=0.3, dirichlet_fraction=0.25
    )
    injector = DirichletInjector()

    policy = torch.tensor([0.5, 0.0, 0.5])
    legal_moves = [0, 2]

    new_policy = injector.inject(policy, legal_moves, config, exploration=True)

    # Policy should have changed but remain a valid probability mass over the legal moves
    assert not torch.allclose(new_policy, policy)
    assert new_policy[1].item() == 0.0  # Illegal move unchanged


def test_action_target_injector():
    """Verifies target actions are boosted correctly during offline re-analysis."""
    config = SimpleNamespace(injection_frac=0.1)
    injector = ActionTargetInjector()

    policy = torch.tensor([0.4, 0.6])
    legal_moves = [0, 1]

    # Inject boost into action 0
    new_policy = injector.inject(policy, legal_moves, config, trajectory_action=0)

    # Original mass is downscaled by (1 - 0.1) = 0.9.
    # 0.4 * 0.9 = 0.36. Boosted by 0.1 = 0.46
    # 0.6 * 0.9 = 0.54.
    assert torch.allclose(new_policy, torch.tensor([0.46, 0.54]))


def test_gumbel_injector_with_raw_policy():
    """Verifies Gumbel noise transforms logits properly while safely ignoring masked actions."""
    torch.manual_seed(42)
    injector = GumbelInjector()

    policy = torch.tensor([0.8, 0.0, 0.2])
    legal_moves = [0, 2]

    new_policy = injector.inject(policy, legal_moves, config=None, exploration=True)

    # Must remain normalized and the illegal move must stay exactly 0
    assert torch.allclose(new_policy.sum(), torch.tensor(1.0))
    assert new_policy[1].item() == 0.0
    assert not torch.allclose(new_policy, policy)


def test_gumbel_injector_with_policy_dist():
    """Verifies Gumbel extraction can directly consume a distribution object to avoid log(prob) instability."""
    torch.manual_seed(42)
    injector = GumbelInjector()

    class DummyDist:
        # A batch of 1
        logits = torch.tensor([[-0.2231, -float("inf"), -1.6094]])

    policy = torch.tensor([0.8, 0.0, 0.2])
    legal_moves = [0, 2]

    new_policy = injector.inject(
        policy, legal_moves, config=None, policy_dist=DummyDist(), exploration=True
    )

    assert torch.allclose(new_policy.sum(), torch.tensor(1.0))
    assert new_policy[1].item() == 0.0
