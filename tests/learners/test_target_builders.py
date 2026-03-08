import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import numpy as np

from agents.learners.target_builders import (
    TargetOutput,
    DQNTargetBuilder,
    PPOTargetBuilder,
    MuZeroTargetBuilder,
    TargetBuilderPipeline,
)
from modules.world_models.inference_output import LearningOutput

pytestmark = pytest.mark.unit


def test_target_output_defaults():
    """Test that TargetOutput initializes with None defaults."""
    output = TargetOutput()
    assert output.q_values is None
    assert output.value_targets is None
    assert output.advantages is None
    assert output.old_log_probs is None
    assert output.policies is None
    assert output.rewards is None
    assert output.chance_codes is None
    assert output.consistency_targets is None
    assert output.target_dist is None
    assert output.values is None
    assert output.chance_values is None
    assert output.to_plays is None
    assert output.action_mask is None
    assert output.obs_mask is None
    assert output.dones is None


def test_dqn_target_builder_standard(rainbow_config):
    """Test standard DQN Bellman target calculation."""
    torch.manual_seed(42)
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

    assert isinstance(targets, TargetOutput)
    # next_q_values.argmax(dim=-1) -> [1, 1]
    # target_q_values[:, [1, 1]] -> [200.0, 400.0]
    # target_q[0] = 1.0 + 0.9 * (1 - False) * 200.0 = 181.0
    # target_q[1] = 0.0 + 0.9 * (1 - True) * 400.0 = 0.0

    expected_q = torch.tensor([181.0, 0.0])
    assert torch.allclose(targets.q_values, expected_q)
    assert torch.allclose(targets.values, expected_q)


def test_dqn_target_builder_c51(rainbow_config):
    """Test C51 distributional Bellman target calculation."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    rainbow_config.atom_size = 3
    rainbow_config.v_min = 0.0
    rainbow_config.v_max = 2.0
    rainbow_config.discount_factor = 1.0
    rainbow_config.n_step = 1

    builder = DQNTargetBuilder(rainbow_config, device)

    # support is [0.0, 1.0, 2.0]
    batch = {
        "rewards": torch.tensor([0.5]),
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
    assert isinstance(targets, TargetOutput)
    target_dist = targets.target_dist

    # Tz = 0.5 + 1.0 * [0.0, 1.0, 2.0] = [0.5, 1.5, 2.5] -> clamp [0.5, 1.5, 2.0]
    # b = 1.5, l = 1, u = 2, dist_l = 0.5, dist_u = 0.5
    expected_dist = torch.tensor([[0.0, 0.5, 0.5]])
    assert torch.allclose(target_dist, expected_dist, atol=1e-3)
    assert targets.values is not None


def test_muzero_target_builder_mapping(muzero_config):
    """Test MuZero target mapping."""
    device = torch.device("cpu")
    builder = MuZeroTargetBuilder()

    batch = {
        "target_values": torch.randn(2, 5),
        "target_policies": torch.randn(2, 5, 4),
        "target_rewards": torch.randn(2, 5),
        "action_mask": torch.ones(2, 5).bool(),
        "obs_mask": torch.ones(2, 5).bool(),
        "dones": torch.zeros(2, 5).bool(),
        "chance_codes": torch.randint(0, 5, (2, 5)),
        "consistency_targets": torch.randn(2, 5, 128),
    }

    targets = builder.build_targets(batch, LearningOutput(), MagicMock())
    assert isinstance(targets, TargetOutput)

    assert torch.allclose(targets.values, batch["target_values"])
    assert torch.allclose(targets.policies, batch["target_policies"])
    assert torch.allclose(targets.rewards, batch["target_rewards"])
    assert torch.allclose(targets.action_mask, batch["action_mask"])
    assert torch.allclose(targets.obs_mask, batch["obs_mask"])
    assert torch.allclose(targets.dones, batch["dones"])
    assert torch.allclose(targets.chance_codes, batch["chance_codes"])
    assert torch.allclose(targets.consistency_targets, batch["consistency_targets"])


def test_ppo_target_builder_gae(ppo_config):
    """Test PPO GAE target calculation."""
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    ppo_config.discount_factor = 0.9
    ppo_config.gae_lambda = 0.5

    builder = PPOTargetBuilder(ppo_config, device)

    batch = {
        "rewards": torch.tensor([1.0, 2.0]),
        "dones": torch.tensor([False, False]),
        "log_probs": torch.tensor([-0.5, -0.6]),
    }
    predictions = LearningOutput(
        values=torch.tensor([10.0, 20.0, 30.0]),
    )

    targets = builder.build_targets(batch, predictions, MagicMock())
    assert isinstance(targets, TargetOutput)

    # d0 = 1.0 + 0.9 * 20.0 - 10.0 = 9.0
    # d1 = 2.0 + 0.9 * 30.0 - 20.0 = 9.0
    # a1 = 9.0, a0 = 9.0 + 0.45 * 9.0 = 13.05
    expected_advantages = torch.tensor([13.05, 9.0])
    assert torch.allclose(targets.advantages, expected_advantages)
    assert torch.allclose(targets.old_log_probs, batch["log_probs"])


def test_ppo_target_builder_precomputed(ppo_config):
    """Test PPO target builder with precomputed advantages."""
    device = torch.device("cpu")
    builder = PPOTargetBuilder(ppo_config, device)

    batch = {
        "advantages": torch.tensor([1.0]),
        "returns": torch.tensor([2.0]),
        "log_probs": torch.tensor([-0.1]),
    }
    targets = builder.build_targets(batch, LearningOutput(), MagicMock())
    assert targets.advantages[0] == 1.0
    assert targets.value_targets[0] == 2.0
    assert targets.old_log_probs[0] == -0.1


def test_target_builder_pipeline():
    """Test that TargetBuilderPipeline merges fields correctly."""

    class MockBuilder1(BaseTargetBuilder):
        def build_targets(self, b, p, n):
            return TargetOutput(q_values=torch.tensor([1.0]))

    class MockBuilder2(BaseTargetBuilder):
        def build_targets(self, b, p, n):
            return TargetOutput(value_targets=torch.tensor([2.0]))

    pipeline = TargetBuilderPipeline([MockBuilder1(), MockBuilder2()])
    targets = pipeline.build_targets({}, LearningOutput(), MagicMock())

    assert targets.q_values[0] == 1.0
    assert targets.value_targets[0] == 2.0


def test_dqn_target_builder_truncated(rainbow_config):
    """Test DQN builder with truncate bootstrapping."""
    device = torch.device("cpu")
    rainbow_config.bootstrap_on_truncated = True
    rainbow_config.discount_factor = 0.9
    rainbow_config.n_step = 1

    builder = DQNTargetBuilder(rainbow_config, device)

    batch = {
        "rewards": torch.tensor([1.0, 1.0]),
        "dones": torch.tensor([True, True]),
        "terminated": torch.tensor([True, False]),
    }
    predictions = LearningOutput(
        next_q_values=torch.tensor([[10.0, 10.0], [10.0, 10.0]]),
        target_q_values=torch.tensor([[100.0, 100.0], [100.0, 100.0]]),
    )

    targets = builder.build_targets(batch, predictions, MagicMock())
    # Index 0: 1.0 + 0 = 1.0
    # Index 1: 1.0 + 0.9 * 100.0 = 91.0
    expected_q = torch.tensor([1.0, 91.0])
    assert torch.allclose(targets.q_values, expected_q)
