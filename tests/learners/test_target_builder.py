import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import numpy as np

from agents.learners.target_builder import (
    DQNTargetBuilder,
    PPOTargetBuilder,
    MuZeroTargetBuilder,
)

from modules.world_models.inference_output import LearningOutput

pytestmark = pytest.mark.unit


def test_dqn_target_builder_standard(rainbow_config):
    """Test standard DQN Bellman target calculation."""
    device = torch.device("cpu")
    rainbow_config.atom_size = 1
    rainbow_config.discount_factor = 0.9
    rainbow_config.n_step = 1

    builder = DQNTargetBuilder(rainbow_config, device)

    batch = {
        "rewards": torch.tensor([1.0, 0.0]),
        "dones": torch.tensor([False, True]),
    }
    predictions = LearningOutput(
        next_q_values=torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        target_q_values=torch.tensor([[100.0, 200.0], [300.0, 400.0]]),
    )

    targets = builder.build_targets(batch, predictions, MagicMock())

    # next_q_values.argmax(dim=-1) -> [1, 1]
    # target_q_values[:, [1, 1]] -> [200.0, 400.0]
    # target_q[0] = 1.0 + 0.9 * (1 - False) * 200.0 = 181.0
    # target_q[1] = 0.0 + 0.9 * (1 - True) * 400.0 = 0.0

    expected_q = torch.tensor([181.0, 0.0])
    assert torch.allclose(targets["q_values"], expected_q)


def test_dqn_target_builder_c51_standard(rainbow_config):
    """Test C51 distributional Bellman target calculation."""
    device = torch.device("cpu")
    rainbow_config.atom_size = 3
    rainbow_config.v_min = 0.0
    rainbow_config.v_max = 2.0
    rainbow_config.discount_factor = 1.0  # Simplify math
    rainbow_config.n_step = 1

    builder = DQNTargetBuilder(rainbow_config, device)

    # support is [0.0, 1.0, 2.0]

    batch = {
        "rewards": torch.tensor(
            [0.5]
        ),  # shifts support to [0.5, 1.5, 2.5] -> clamped [0.5, 1.5, 2.0]
        "dones": torch.tensor([False]),
    }

    # online next q: forces max action index 0
    next_q_logits = torch.tensor([[10.0, 0.0, 0.0]])

    # target next distribution: all mass at support index 1 (value=1.0)
    target_q_logits = torch.tensor(
        [
            [
                [0.0, 10.0, 0.0],  # Action 0
                [0.0, 0.0, 0.0],  # Action 1
                [0.0, 0.0, 0.0],  # Action 2
            ]
        ]
    )

    predictions = LearningOutput(
        next_q_logits=next_q_logits,
        target_q_logits=target_q_logits,
    )

    targets = builder.build_targets(batch, predictions, MagicMock())
    target_dist = targets["target_dist"]

    # Tz = 0.5 + 1.0 * [0.0, 1.0, 2.0] = [0.5, 1.5, 2.5] -> clamp [0.5, 1.5, 2.0]
    # For action 0, prob is [0.0, 1.0, 0.0] (mass at support[1]=1.0)
    # T(support[1]) = 0.5 + 1.0 = 1.5
    # b = (1.5 - 0.0) / ((2.0 - 0.0) / (3 - 1)) = 1.5 / 1.0 = 1.5
    # l = 1, u = 2
    # dist_l = 2 - 1.5 = 0.5
    # dist_u = 1.5 - 1 = 0.5
    # projected mass at index 1: 1.0 * 0.5 = 0.5
    # projected mass at index 2: 1.0 * 0.5 = 0.5

    expected_dist = torch.tensor([[0.0, 0.5, 0.5]])
    assert torch.allclose(target_dist, expected_dist, atol=1e-3)


def test_ppo_target_builder_gae_standard(ppo_config):
    """Test PPO GAE target calculation."""
    device = torch.device("cpu")
    ppo_config.discount_factor = 0.9
    ppo_config.gae_lambda = 0.5

    builder = PPOTargetBuilder(ppo_config, device)

    # 2 transitions
    batch = {
        "rewards": torch.tensor([1.0, 2.0]),
        "dones": torch.tensor([False, False]),
    }
    # Values for s0, s1, s2
    predictions = LearningOutput(
        values=torch.tensor([10.0, 20.0, 30.0]),
    )

    targets = builder.build_targets(batch, predictions, MagicMock())

    # deltas:
    # d0 = r0 + gamma * v1 - v0 = 1.0 + 0.9 * 20.0 - 10.0 = 1.0 + 18.0 - 10.0 = 9.0
    # d1 = r1 + gamma * v2 - v1 = 2.0 + 0.9 * 30.0 - 20.0 = 2.0 + 27.0 - 20.0 = 9.0

    # advantages (discounted cumulative sum of deltas with gamma * lambda = 0.9 * 0.5 = 0.45):
    # a1 = d1 = 9.0
    # a0 = d0 + 0.45 * a1 = 9.0 + 0.45 * 9.0 = 9.0 + 4.05 = 13.05

    expected_advantages = torch.tensor([13.05, 9.0])
    assert torch.allclose(targets["advantages"], expected_advantages)


def test_muzero_target_builder_mapping(muzero_config):
    """Test MuZero target mapping."""
    device = torch.device("cpu")
    builder = MuZeroTargetBuilder(muzero_config, device)

    batch = {
        "values": torch.randn(2, 5),
        "policies": torch.randn(2, 5, 4),
        "rewards": torch.randn(2, 5),
        "action_mask": torch.ones(2, 5).bool(),
        "obs_mask": torch.ones(2, 5).bool(),
        "consistency_targets": torch.randn(2, 5, 128),
        "extra_info": "should be ignored",
    }

    targets = builder.build_targets(batch, LearningOutput(), MagicMock())

    assert "values" in targets
    assert "policies" in targets
    assert "rewards" in targets
    assert "action_mask" in targets
    assert "obs_mask" in targets
    assert "consistency_targets" in targets
    assert "extra_info" not in targets
    assert torch.allclose(targets["values"], batch["values"].to(device))
    assert torch.allclose(
        targets["consistency_targets"], batch["consistency_targets"].to(device)
    )


def test_dqn_target_builder_truncated_bootstrapping(rainbow_config):
    """Test DQN target calculation with truncation bootstrapping."""
    device = torch.device("cpu")
    rainbow_config.bootstrap_on_truncated = True
    rainbow_config.discount_factor = 0.9
    rainbow_config.n_step = 1

    builder = DQNTargetBuilder(rainbow_config, device)

    batch = {
        "rewards": torch.tensor([1.0, 1.0]),
        "dones": torch.tensor([True, True]),
        "terminated": torch.tensor([True, False]),
        "truncated": torch.tensor([False, True]),
    }
    predictions = LearningOutput(
        next_q_values=torch.tensor([[10.0, 10.0], [10.0, 10.0]]),
        target_q_values=torch.tensor([[100.0, 100.0], [100.0, 100.0]]),
    )

    targets = builder.build_targets(batch, predictions, MagicMock())

    # Index 0: terminated=True -> bootstrap False
    # target_q[0] = 1.0 + 0 = 1.0
    # Index 1: terminated=False, truncated=True -> bootstrap True (since bootstrap_on_truncated=True)
    # target_q[1] = 1.0 + 0.9 * 1.0 * 100.0 = 91.0

    expected_q = torch.tensor([1.0, 91.0])
    assert torch.allclose(targets["q_values"], expected_q)


def test_ppo_target_builder_padding_values(ppo_config):
    """Test PPO target builder with automatic value padding."""
    device = torch.device("cpu")
    ppo_config.discount_factor = 0.99
    builder = PPOTargetBuilder(ppo_config, device)

    # 1 transition, but values only for s0
    batch = {
        "rewards": torch.tensor([1.0]),
        "dones": torch.tensor([False]),
    }
    predictions = LearningOutput(
        values=torch.tensor([10.0]),
    )

    targets = builder.build_targets(batch, predictions, MagicMock())
    # v_all becomes [10.0, 0.0]
    # delta = 1.0 + 0.99 * 0.0 - 10.0 = -9.0
    # advantage = -9.0
    assert targets["advantages"].shape == (1,)
    assert targets["advantages"][0] == -9.0
