import pytest
import torch
from modules.heads.strategies import (
    ScalarStrategy,
    Categorical,
    MuZeroSupport,
    C51Support,
    GaussianStrategy,
)
from modules.distributions import Deterministic

pytestmark = pytest.mark.unit


def test_scalar_strategy():
    strategy = ScalarStrategy(output_size=2)
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 1.0], [3.0, 4.0]])
    loss = strategy.compute_loss(pred, target)
    assert torch.allclose(loss, torch.tensor([[0.0, 1.0], [0.0, 0.0]]))

    strat_1d = ScalarStrategy(output_size=1)
    out_1d = torch.tensor([[5.0], [6.0]])
    exp_val = strat_1d.to_expected_value(out_1d)
    assert exp_val.shape == (2,)


def test_categorical_strategy():
    strategy = Categorical(num_classes=3)
    pred = torch.tensor([[10.0, 1.0, 1.0]])
    target_idx = torch.tensor([0])
    loss_idx = strategy.compute_loss(pred, target_idx)
    assert loss_idx.shape == (1,)


def test_muzero_support_strategy():
    strategy = MuZeroSupport(support_range=10, eps=0.001)

    # 0.0 maps exactly to the middle bin (index 10)
    scalar = torch.tensor([0.0])
    target = strategy.scalar_to_target(scalar)
    assert target.shape == (1, 21)
    assert torch.allclose(target[0, 10], torch.tensor(1.0))

    logits = target * 100.0
    exp_val = strategy.to_expected_value(logits)
    assert torch.allclose(exp_val, scalar, atol=1e-3)

    loss = strategy.compute_loss(logits, target)
    assert loss.shape == (1,)


def test_c51_support_strategy():
    strategy = C51Support(v_min=-10.0, v_max=10.0, num_atoms=21)
    scalar = torch.tensor([5.5])
    target = strategy.scalar_to_target(scalar)

    assert torch.allclose(target[0, 15], torch.tensor(0.5))
    assert torch.allclose(target[0, 16], torch.tensor(0.5))


def test_gaussian_strategy():
    strategy = GaussianStrategy(action_dim=2, min_log_std=-10.0, max_log_std=2.0)
    pred = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0]])
    loss = strategy.compute_loss(pred, target)
    assert loss.shape == (1,)
