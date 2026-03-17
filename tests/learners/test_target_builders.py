import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import numpy as np

from agents.learners.target_builders import (
    DQNTargetBuilder,
    BaseTargetBuilder,
)

pytestmark = pytest.mark.unit


def test_base_target_builder_instantiation():
    """Verify BaseTargetBuilder cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseTargetBuilder()


def test_dqn_target_builder_standard(rainbow_config):
    """Test standard DQN Bellman target calculation."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    rainbow_config.atom_size = 1
    rainbow_config.discount_factor = 0.9
    rainbow_config.n_step = 1

    builder = DQNTargetBuilder(
        device=device,
        target_network=MagicMock(),
        gamma=rainbow_config.discount_factor,
        n_step=rainbow_config.n_step,
        use_c51=rainbow_config.atom_size > 1,
    )

    # Batch size 2
    batch = {
        "next_observations": torch.randn((2, 4)),
        "rewards": torch.tensor([1.0, 0.0]),
        "dones": torch.tensor([False, True]),
        "actions": torch.tensor([0, 0]),
    }
    
    # Target Builder expects B dimension (it handles T dimension internally if needed)
    predictions = {
        "q_values": torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
    }
    
    # Monitor network call
    network = MagicMock()
    network.learner_inference.return_value = {
        "q_values": torch.tensor([[[0.0, 10.0]], [[0.0, 10.0]]]) # [B=2, T=1, Actions]
    }
    builder.target_network.learner_inference.return_value = {
        "q_values": torch.tensor([[[100.0, 200.0]], [[300.0, 400.0]]]) # [B=2, T=1, Actions]
    }

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

    builder = DQNTargetBuilder(
        device=device,
        target_network=MagicMock(),
        gamma=rainbow_config.discount_factor,
        n_step=rainbow_config.n_step,
        use_c51=rainbow_config.atom_size > 1,
        v_min=rainbow_config.v_min,
        v_max=rainbow_config.v_max,
        atom_size=rainbow_config.atom_size,
    )

    # support is [0.0, 1.0, 2.0]
    batch = {
        "next_observations": torch.randn((1, 4)),
        "rewards": torch.tensor([0.5]),
        "dones": torch.tensor([False]),
        "actions": torch.tensor([0]),
    }

    # Online next q logits: forces max action index 0
    network = MagicMock()
    network.learner_inference.return_value = {
        "q_logits": torch.tensor([[[10.0, 0.0, 0.0]]])
    }
    # target next distribution: all mass at support index 1 (value=1.0)
    builder.target_network.learner_inference.return_value = {
        "q_logits": torch.tensor([[[0.0, 10.0, 0.0], [0.0, 0.0, 0.0]]]) # [B=1, T=1, Actions, Atoms]
    }

    predictions = {
        "q_logits": torch.randn((1, 2, 3)),
    }

    targets = builder.build_targets(batch, predictions, network)

    assert isinstance(targets, dict)
    # Tz = 0.5 + 1.0 * [0.0, 1.0, 2.0] = [0.5, 1.5, 2.5] -> clamp [0.5, 1.5, 2.0]
    # b = 1.5, l = 1, u = 2, dist_l = 0.5, dist_u = 0.5
    expected_probs = torch.tensor([[0.0, 0.5, 0.5]])
    assert torch.allclose(targets["q_logits"], expected_probs, atol=1e-3)




def test_dqn_target_builder_truncated(rainbow_config):
    """Test target calculation when bootstrap_on_truncated is enabled."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    rainbow_config.atom_size = 1
    rainbow_config.discount_factor = 0.9
    rainbow_config.n_step = 1
    rainbow_config.bootstrap_on_truncated = True

    builder = DQNTargetBuilder(
        device=device,
        target_network=MagicMock(),
        gamma=rainbow_config.discount_factor,
        n_step=rainbow_config.n_step,
        use_c51=rainbow_config.atom_size > 1,
        bootstrap_on_truncated=rainbow_config.bootstrap_on_truncated,
    )

    batch = {
        "next_observations": torch.randn((2, 4)),
        "rewards": torch.tensor([1.0, 1.0]),
        "dones": torch.tensor([False, False]), # truncated is False (online)
        "actions": torch.tensor([0, 0]),
    }
    
    batch["terminated"] = torch.tensor([True, False])

    network = MagicMock()
    network.learner_inference.return_value = {
        "q_values": torch.tensor([[[10.0, 10.0]], [[10.0, 10.0]]]),
    }
    
    builder.target_network.learner_inference.return_value = {
        "q_values": torch.tensor([[[100.0, 100.0]], [[100.0, 100.0]]])
    }

    predictions = {
        "q_values": torch.tensor([[10.0, 10.0], [10.0, 10.0]]),
    }

    targets = builder.build_targets(batch, predictions, network)

    assert isinstance(targets, dict)
    # Sample 0: Terminated=True -> target = reward = 1.0
    # Sample 1: Terminated=False -> target = reward + gamma * max_next_q = 1.0 + 0.9 * 100.0 = 91.0
    expected_q = torch.tensor([1.0, 91.0])
    assert torch.allclose(targets["q_values"], expected_q)
