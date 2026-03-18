import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import numpy as np

from agents.learner.target_builders import (
    TemporalDifferenceBuilder,
    LatentConsistencyBuilder,
    TrajectoryGradientScaleBuilder,
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

    builder = TemporalDifferenceBuilder(
        target_network=MagicMock(),
        gamma=rainbow_config.discount_factor,
        n_step=rainbow_config.n_step,
        bootstrap_on_truncated=getattr(rainbow_config, "bootstrap_on_truncated", False),
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
        "q_values": torch.tensor([[[0.0, 10.0]], [[0.0, 10.0]]])  # [B=2, T=1, Actions]
    }
    builder.target_network.learner_inference.return_value = {
        "q_values": torch.tensor(
            [[[100.0, 200.0]], [[300.0, 400.0]]]
        )  # [B=2, T=1, Actions]
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



def test_dqn_target_builder_truncated(rainbow_config):
    """Test target calculation when bootstrap_on_truncated is enabled."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    rainbow_config.atom_size = 1
    rainbow_config.discount_factor = 0.9
    rainbow_config.n_step = 1
    rainbow_config.bootstrap_on_truncated = True

    builder = TemporalDifferenceBuilder(
        target_network=MagicMock(),
        gamma=rainbow_config.discount_factor,
        n_step=rainbow_config.n_step,
        bootstrap_on_truncated=rainbow_config.bootstrap_on_truncated,
    )

    batch = {
        "next_observations": torch.randn((2, 4)),
        "rewards": torch.tensor([1.0, 1.0]),
        "dones": torch.tensor([False, False]),  # truncated is False (online)
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


def test_latent_consistency_builder():
    """Test LatentConsistencyBuilder for EfficientZero targets."""
    torch.manual_seed(42)
    builder = LatentConsistencyBuilder()

    batch = {
        "unroll_observations": torch.randn((2, 3, 4, 8, 8)),  # [B, T+1, C, H, W]
    }

    network = MagicMock()
    # Mock network.obs_inference returning InferenceOutput with network_state.dynamics
    initial_out = MagicMock()
    initial_out.network_state.dynamics = torch.randn((6, 16))  # [B*(T+1), Latent]
    network.obs_inference.return_value = initial_out

    # Mock network.project returning projects embeddings
    network.project.return_value = torch.randn((6, 32))  # [B*(T+1), Proj]

    targets = builder.build_targets(batch, {}, network)

    assert "consistency_targets" in targets
    assert targets["consistency_targets"].shape == (2, 3, 32)
    assert torch.allclose(
        torch.norm(targets["consistency_targets"], p=2, dim=-1), torch.ones((2, 3))
    )


def test_trajectory_gradient_scale_builder():
    """Test TrajectoryGradientScaleBuilder for BPTT scaling."""
    builder = TrajectoryGradientScaleBuilder(unroll_steps=5)
    batch = {"rewards": torch.zeros(1)}  # dummy to provide device

    targets = builder.build_targets(batch, {}, MagicMock())

    assert "gradient_scales" in targets
    expected_scales = torch.tensor([[1.0, 0.2, 0.2, 0.2, 0.2, 0.2]])
    assert torch.allclose(targets["gradient_scales"], expected_scales)
