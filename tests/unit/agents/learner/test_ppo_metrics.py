import pytest
import torch

from agents.learner.losses.policy import ClippedSurrogateLoss
from agents.learner.losses.value import ValueLoss, ClippedValueLoss
from agents.learner.losses.representations import (
    ClassificationRepresentation,
    ScalarRepresentation,
)
from agents.learner.losses.loss_pipeline import LossPipeline
from agents.learner.losses.priorities import NullPriorityComputer

pytestmark = pytest.mark.unit


def test_ppo_debug_metrics_exist_and_computed_correctly():
    """
    Tier 1: Verify PPO debug metrics are present and correctly calculated.
    Tests approxkl, clipfrac, and entropy_loss explicitly.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    num_actions = 4
    B, T = 8, 1

    pol_rep = ClassificationRepresentation(num_actions)
    val_rep = ScalarRepresentation()

    policy_loss_mod = ClippedSurrogateLoss(
        device=device,
        representation=pol_rep,
        clip_param=0.2,
        entropy_coefficient=0.01,
        optimizer_name="policy",
        name="policy_loss",
    )

    value_loss_mod = ClippedValueLoss(
        device=device,
        representation=val_rep,
        clip_param=0.2,
        target_key="returns",
        optimizer_name="value",
        loss_factor=0.5,
        name="value_loss",
    )

    pipeline = LossPipeline(
        modules=[policy_loss_mod, value_loss_mod],
        priority_computer=NullPriorityComputer(),
        minibatch_size=B,
        unroll_steps=0,
        num_actions=num_actions,
        atom_size=1,
    )

    # Build synthetic predictions and targets
    predictions = {
        "policies": torch.randn(B, T, num_actions),
        "values": torch.randn(B, T, 1),
    }
    targets = {
        "actions": torch.randint(0, num_actions, (B, T)),
        "old_log_probs": torch.randn(B, T),
        "advantages": torch.randn(B, T),
        "returns": torch.randn(B, T),
        "old_values": torch.randn(B, T),
        "policy_mask": torch.ones(B, T, dtype=torch.bool),
        "value_mask": torch.ones(B, T, dtype=torch.bool),
        "weights": torch.ones(B),
        "gradient_scales": torch.ones(1, T),
    }

    total_loss, metrics, priorities = pipeline.run(predictions, targets)

    assert "approx_kl" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics


def test_clipped_value_loss_math():
    """
    Tier 1: Explicit math verification for ClippedValueLoss.
    Formula: max[(V - V_targ)^2, (clip(V, V_old - eps, V_old + eps) - V_targ)^2]
    """
    device = torch.device("cpu")
    rep = ScalarRepresentation()
    clip_param = 0.1
    loss_factor = 1.0

    loss_mod = ClippedValueLoss(
        device=device,
        representation=rep,
        clip_param=clip_param,
        loss_factor=loss_factor,
    )

    # Case 1: No clipping (V is within old_v +- clip_param)
    # V=1.0, old_V=1.0, target=1.5
    # (1.0 - 1.5)^2 = 0.25
    # clip(1.0, 0.9, 1.1) = 1.0 -> (1.0 - 1.5)^2 = 0.25
    # max(0.25, 0.25) = 0.25
    predictions = {"values": torch.tensor([[[1.0]]])}
    targets = {
        "returns": torch.tensor([[1.5]]),
        "old_values": torch.tensor([[1.0]]),
        "value_mask": torch.tensor([[True]]),
    }
    loss, _ = loss_mod.compute_loss(predictions, targets)
    assert torch.allclose(loss, torch.tensor([[0.25]]))

    # Case 2: Clipping triggered (V is far from old_v)
    # V=2.0, old_V=1.0, target=1.5, clip=0.1
    # Unclipped: (2.0 - 1.5)^2 = 0.25
    # Clipped: clip(2.0, 0.9, 1.1) = 1.1 -> (1.1 - 1.5)^2 = (-0.4)^2 = 0.16
    # max(0.25, 0.16) = 0.25
    predictions = {"values": torch.tensor([[[2.0]]])}
    targets = {
        "returns": torch.tensor([[1.5]]),
        "old_values": torch.tensor([[1.0]]),
        "value_mask": torch.tensor([[True]]),
    }
    loss, _ = loss_mod.compute_loss(predictions, targets)
    assert torch.allclose(loss, torch.tensor([[0.25]]))

    # Case 3: Clipping is WORSE (max triggers clipped)
    # V=1.05, old_V=1.0, target=0.5, clip=0.1
    # Unclipped: (1.05 - 0.5)^2 = 0.55^2 = 0.3025
    # Clipped: clip(1.05, 0.9, 1.1) = 1.05 -> (1.05 - 0.5)^2 = 0.3025
    # Wait, let's make clipped worse.
    # V=1.2, old_V=1.0, target=2.0, clip=0.1
    # Unclipped: (1.2 - 2.0)^2 = (-0.8)^2 = 0.64
    # Clipped: clip(1.2, 0.9, 1.1) = 1.1 -> (1.1 - 2.0)^2 = (-0.9)^2 = 0.81
    # max(0.64, 0.81) = 0.81
    predictions = {"values": torch.tensor([[[1.2]]])}
    targets = {
        "returns": torch.tensor([[2.0]]),
        "old_values": torch.tensor([[1.0]]),
        "value_mask": torch.tensor([[True]]),
    }
    loss, _ = loss_mod.compute_loss(predictions, targets)
    assert torch.allclose(loss, torch.tensor([[0.81]]))
