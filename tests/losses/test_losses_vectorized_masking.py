import pytest

pytestmark = pytest.mark.unit

import torch
import torch.nn.functional as F
from losses.losses import (
    LossPipeline,
    ValueLoss,
    PolicyLoss,
    RewardLoss,
    ConsistencyLoss,
)
from agents.learners.target_builders import TargetOutput
from modules.world_models.inference_output import LearningOutput
from types import SimpleNamespace


def test_vectorized_loss_masking():
    """
    Test that vectorized loss calculation matches expectations and respects masks.
    """
    device = "cpu"
    batch_size = 2
    unroll_steps = 3
    t_plus_1 = unroll_steps + 1
    num_actions = 4

    config = SimpleNamespace(
        minibatch_size=batch_size,
        unroll_steps=unroll_steps,
        support_range=None,
        value_loss_factor=1.0,
        value_loss_function=F.mse_loss,
        policy_loss_function=F.cross_entropy,
        reward_loss_function=F.mse_loss,
        consistency_loss_factor=1.0,
        mask_absorbing=True,
        game=SimpleNamespace(num_players=1),
    )

    # 1. Setup Predictions and Targets with natural lengths
    # Values/Policies: (B, T+1, ...)
    # Rewards/Actions: (B, T, ...)

    predictions = LearningOutput(
        values=torch.ones((batch_size, t_plus_1, 1), device=device) * 2.0,
        policies=torch.randn((batch_size, t_plus_1, num_actions), device=device),
        rewards=torch.ones((batch_size, unroll_steps, 1), device=device) * 0.5,
        latents=torch.randn((batch_size, t_plus_1, 64), device=device),
    )

    targets = TargetOutput(
        values=torch.ones((batch_size, t_plus_1), device=device) * 1.0,
        returns=torch.ones((batch_size, t_plus_1), device=device) * 1.0,
        policies=F.softmax(
            torch.randn((batch_size, t_plus_1, num_actions), device=device), dim=-1
        ),
        rewards=torch.ones((batch_size, unroll_steps), device=device) * 1.0,
        consistency_targets=torch.randn((batch_size, t_plus_1, 64), device=device),
    )

    # 2. Setup Masks
    # Batch 0: Full sequence valid
    # Batch 1: Terminal at t=1 (only s0, s1 valid)
    obs_mask = torch.tensor(
        [[True, True, True, True], [True, True, False, False]], device=device
    )

    # action_mask: False for terminal state (t=1 for batch 1)
    action_mask = torch.tensor(
        [[True, True, True, True], [True, False, False, False]], device=device
    )

    same_game = torch.ones((batch_size, t_plus_1), dtype=torch.bool, device=device)
    context = {
        "has_valid_obs_mask": obs_mask,
        "has_valid_action_mask": action_mask,
        "is_same_game": same_game,
    }

    weights = torch.ones(batch_size, device=device)
    gradient_scales = torch.ones((1, t_plus_1), device=device)

    # Mock Model for ConsistencyLoss
    class MockModel:
        def project(self, x, grad=True):
            return x

    mock_model = MockModel()

    # 3. Create modules
    modules = [
        ValueLoss(config, device),
        PolicyLoss(config, device),
        RewardLoss(config, device),
        ConsistencyLoss(config, device, mock_model),
    ]
    pipeline = LossPipeline(modules)

    # 4. Run Pipeline
    loss_mean, loss_dict, priorities = pipeline.run(
        predictions=predictions,
        targets=targets,
        context=context,
        weights=weights,
        gradient_scales=gradient_scales,
    )

    # 5. Assertions
    # Value Loss check for Batch 1: Now counts all 4 steps because it uses is_same_game.
    # (2.0 - 1.0)^2 = 1.0 per step.
    # Batch 0: 4 steps * 1.0 = 4.0
    # Batch 1: 4 steps * 1.0 = 4.0
    # Total = 8.0 / (batch_size=2) = 4.0
    assert pytest.approx(loss_dict["ValueLoss"]) == 4.0

    # Reward Loss check for Batch 1: Now counts all 3 steps because it uses is_same_game.
    # (0.5 - 1.0)^2 = 0.25 per step.
    # Batch 0: 3 steps * 0.25 = 0.75
    # Batch 1: 3 steps * 0.25 = 0.75
    # Total = 1.5 / 2 = 0.75
    assert pytest.approx(loss_dict["RewardLoss"]) == 0.75

    # Policy Loss check for Batch 1: action_mask = [T, F, F, F].
    # Only k=0 is valid for batch 1.
    # Batch 0: 4 steps. Batch 1: 1 step.
    # Total 5 steps of categorical cross entropy.
    assert "PolicyLoss" in loss_dict
    assert loss_dict["PolicyLoss"] > 0
