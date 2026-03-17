import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import numpy as np

from agents.learners.target_builders import (
    DQNTargetBuilder,
    PPOTargetBuilder,
    MuZeroTargetBuilder,
    BaseTargetBuilder,
)
from modules.world_models.inference_output import LearningOutput

pytestmark = pytest.mark.unit


def test_base_target_builder_instantiation():
    """Verify BaseTargetBuilder cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseTargetBuilder(MagicMock(), torch.device("cpu"))


def test_dqn_target_builder_standard(rainbow_config):
    """Test standard DQN Bellman target calculation."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    rainbow_config.atom_size = 1
    rainbow_config.discount_factor = 0.9
    rainbow_config.n_step = 1

    builder = DQNTargetBuilder(rainbow_config, device, target_network=MagicMock())

    # Batch size 2
    batch = {
        "next_observations": torch.randn((2, 4)),
        "rewards": torch.tensor([1.0, 0.0]),
        "dones": torch.tensor([False, True]),
        "actions": torch.tensor([0, 0]),
    }
    
    # Target Builder expects B dimension (it handles T dimension internally if needed)
    predictions = LearningOutput(
        q_values=torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
    )
    
    # Monitor network call
    network = MagicMock()
    network.learner_inference.return_value = LearningOutput(
        q_values=torch.tensor([[[0.0, 10.0]], [[0.0, 10.0]]]) # [B=2, T=1, Actions]
    )
    builder.target_network.learner_inference.return_value = LearningOutput(
        q_values=torch.tensor([[[100.0, 200.0]], [[300.0, 400.0]]]) # [B=2, T=1, Actions]
    )

    targets = builder.build_targets(batch, predictions, network)

    assert isinstance(targets, dict)
    # Online max action is 1 for both.
    # Target values at index 1: 200.0, 400.0
    # target_q[0] = 1.0 + 0.9 * 200.0 = 181.0
    # target_q[1] = 0.0 + 0.9 * 0.0 = 0.0

    expected_q = torch.tensor([181.0, 0.0])
    assert torch.allclose(targets["q_values"], expected_q)
    assert "actions" in targets


def test_dqn_target_builder_c51(rainbow_config):
    """Test C51 distributional Bellman target calculation."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    rainbow_config.atom_size = 3
    rainbow_config.v_min = 0.0
    rainbow_config.v_max = 2.0
    rainbow_config.discount_factor = 1.0
    rainbow_config.n_step = 1

    builder = DQNTargetBuilder(rainbow_config, device, target_network=MagicMock())

    # support is [0.0, 1.0, 2.0]
    batch = {
        "next_observations": torch.randn((1, 4)),
        "rewards": torch.tensor([0.5]),
        "dones": torch.tensor([False]),
        "actions": torch.tensor([0]),
    }

    # Online next q logits: forces max action index 0
    network = MagicMock()
    network.learner_inference.return_value = LearningOutput(
        q_logits=torch.tensor([[[10.0, 0.0, 0.0]]])
    )

    # target next distribution: all mass at support index 1 (value=1.0)
    builder.target_network.learner_inference.return_value = LearningOutput(
        q_logits=torch.tensor([[[0.0, 10.0, 0.0], [0.0, 0.0, 0.0]]]) # [B=1, T=1, Actions, Atoms]
    )

    predictions = LearningOutput(
        q_logits=torch.randn((1, 2, 3)),
    )

    targets = builder.build_targets(batch, predictions, network)

    assert isinstance(targets, dict)
    # Tz = 0.5 + 1.0 * [0.0, 1.0, 2.0] = [0.5, 1.5, 2.5] -> clamp [0.5, 1.5, 2.0]
    # b = 1.5, l = 1, u = 2, dist_l = 0.5, dist_u = 0.5
    expected_probs = torch.tensor([[0.0, 0.5, 0.5]])
    assert torch.allclose(targets["q_logits"], expected_probs, atol=1e-3)


def test_ppo_target_builder_gae(ppo_config):
    """Test Generalized Advantage Estimation calculation."""
    device = torch.device("cpu")
    ppo_config.discount_factor = 0.9
    ppo_config.gae_lambda = 0.5
    builder = PPOTargetBuilder(ppo_config, device)

    # PPO Target Builder is an adapter that expects pre-computed tags
    # It does NOT compute GAE itself; that's done in Replay Buffer Processors
    batch = {
        "rewards": torch.tensor([1.0, 2.0]),
        "advantages": torch.tensor([13.05, 9.0]),
        "returns": torch.tensor([10.0, 20.0]),
        "dones": torch.tensor([False, False]),
        "log_probabilities": torch.tensor([-0.5, -0.6]),
        "actions": torch.tensor([0, 1]),
    }
    # Values for t=0, 1, 2
    predictions = LearningOutput(
        values=torch.tensor([10.0, 20.0, 30.0]),
    )

    targets = builder.build_targets(batch, predictions, MagicMock())

    assert isinstance(targets, dict)
    assert torch.allclose(targets["advantages"], batch["advantages"])
    assert torch.allclose(targets["returns"], batch["returns"])
    assert torch.allclose(targets["old_log_probs"], batch["log_probabilities"])


def test_ppo_target_builder_precomputed(ppo_config):
    """Test passing precomputed PPO targets."""
    device = torch.device("cpu")
    builder = PPOTargetBuilder(ppo_config, device)

    batch = {
        "advantages": torch.tensor([1.0]),
        "returns": torch.tensor([2.0]),
        "log_probabilities": torch.tensor([-0.1]),
        "actions": torch.tensor([0]),
    }
    targets = builder.build_targets(batch, LearningOutput(), MagicMock())
    assert isinstance(targets, dict)
    assert targets["advantages"][0] == 1.0
    assert targets["returns"][0] == 2.0
    assert targets["old_log_probs"][0] == -0.1


def test_dqn_target_builder_truncated(rainbow_config):
    """Test target calculation when bootstrap_on_truncated is enabled."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    rainbow_config.atom_size = 1
    rainbow_config.discount_factor = 0.9
    rainbow_config.n_step = 1
    rainbow_config.bootstrap_on_truncated = True

    builder = DQNTargetBuilder(rainbow_config, device, target_network=MagicMock())

    batch = {
        "next_observations": torch.randn((2, 4)),
        "rewards": torch.tensor([1.0, 1.0]),
        "dones": torch.tensor([False, False]), # truncated is False (online)
        "actions": torch.tensor([0, 0]),
    }
    
    batch["terminated"] = torch.tensor([True, False])

    network = MagicMock()
    network.learner_inference.return_value = LearningOutput(
        q_values=torch.tensor([[[10.0, 10.0]], [[10.0, 10.0]]]),
    )
    
    builder.target_network.learner_inference.return_value = LearningOutput(
        q_values=torch.tensor([[[100.0, 100.0]], [[100.0, 100.0]]])
    )

    predictions = LearningOutput(
        q_values=torch.tensor([[10.0, 10.0], [10.0, 10.0]]),
    )

    targets = builder.build_targets(batch, predictions, network)

    assert isinstance(targets, dict)
    # Sample 0: Terminated=True -> target = reward = 1.0
    # Sample 1: Terminated=False -> target = reward + gamma * max_next_q = 1.0 + 0.9 * 100.0 = 91.0
    expected_q = torch.tensor([1.0, 91.0])
    assert torch.allclose(targets["q_values"], expected_q)
