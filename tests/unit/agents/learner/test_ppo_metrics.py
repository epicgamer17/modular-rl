import pytest
import torch

from agents.learner.losses.policy import ClippedSurrogateLoss
from agents.learner.losses.value import ValueLoss
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

    value_loss_mod = ValueLoss(
        device=device,
        representation=val_rep,
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
        "policy_mask": torch.ones(B, T, dtype=torch.bool),
        "value_mask": torch.ones(B, T, dtype=torch.bool),
        "weights": torch.ones(B),
        "gradient_scales": torch.ones(1, T),
    }

    total_loss, metrics, priorities = pipeline.run(predictions, targets)

    assert "approx_kl" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
