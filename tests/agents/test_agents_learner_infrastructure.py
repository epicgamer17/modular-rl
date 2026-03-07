import pytest
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

from agents.learners.ppo_learner import PPOLearner
from configs.agents.ppo import PPOConfig
from agents.learners.callbacks import MetricsCallback
from modules.agent_nets.modular import ModularAgentNetwork
from stats.stats import StatTracker

pytestmark = pytest.mark.unit


def test_learner_save_load_checkpoint(cartpole_game_config, tmp_path):
    """
    Test that the learner can save and load checkpoints correctly,
    restoring weights and training steps.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Setup config and learner
    save_dir = tmp_path / "checkpoints"
    save_dir.mkdir()

    config_dict = {
        "steps_per_epoch": 16,
        "num_minibatches": 2,
        "train_policy_iterations": 1,
        "train_value_iterations": 1,
        "learning_rate": 1e-3,
        "clip_param": 0.2,
        "discount_factor": 0.99,
        "gae_lambda": 0.95,
        "entropy_coefficient": 0.01,
        "critic_coefficient": 0.5,
        "action_selector": {"base": {"type": "categorical"}},
        "clipnorm": 0.5,
        "training_steps": 100,
        "results_path": "results",
        "actor_config": {"clipnorm": 0.5},
        "critic_config": {"clipnorm": 0.5},
    }
    config = PPOConfig(config_dict, cartpole_game_config)

    device = torch.device("cpu")
    obs_dim = torch.Size((4,))
    num_actions = 2

    network = ModularAgentNetwork(
        config=config, input_shape=obs_dim, num_actions=num_actions
    ).to(device)

    learner = PPOLearner(
        config=config,
        agent_network=network,
        device=device,
        num_actions=num_actions,
        observation_dimensions=obs_dim,
        observation_dtype=torch.float32,
    )

    learner.training_step = 123
    checkpoint_path = save_dir / "checkpoint.pt"

    # 2. Save checkpoint
    learner.save_checkpoint(str(checkpoint_path))
    assert checkpoint_path.exists(), f"Checkpoint file not found at {checkpoint_path}"

    # 3. Modify weights slightly
    original_weights = {
        k: v.clone() for k, v in learner.agent_network.state_dict().items()
    }
    with torch.no_grad():
        for param in learner.agent_network.parameters():
            param.add_(1.0)

    # Verify weights are different
    for k, v in learner.agent_network.state_dict().items():
        assert not torch.allclose(
            v, original_weights[k]
        ), f"Weight {k} was not modified"

    # 4. Load checkpoint
    learner.load_checkpoint(str(checkpoint_path))

    # Verify weights are restored
    for k, v in learner.agent_network.state_dict().items():
        assert torch.allclose(
            v, original_weights[k]
        ), f"Weight {k} was not restored correctly"

    # Verify step is restored
    assert (
        learner.training_step == 123
    ), f"Expected training_step 123, got {learner.training_step}"


def test_metrics_callback_smoke(make_cartpole_config):
    """
    Test the MetricsCallback by manually firing it with dummy data.
    Ensures it processes data without errors.
    """
    config = make_cartpole_config(
        stochastic=True, latent_viz_interval=1, latent_viz_method="pca"
    )

    callback = MetricsCallback()

    # Dummy learner-like object
    class MockLearner:
        def __init__(self):
            self.config = config
            self.device = torch.device("cpu")
            self.training_step = 0

    learner = MockLearner()
    stats = StatTracker(name="test")

    # Dummy data
    predictions = {
        "latent_states": torch.randn(8, 2, 128),  # [B, T+1, D]
        "chance_codes": torch.randn(8, 2, 1, 10),  # [B, T+1, 1, K]
    }
    targets = {
        "actions": torch.zeros(8, 2, 1).long(),
        "chance_codes": torch.zeros(8, 2, 1).long(),
    }

    # Fire callback
    callback.on_step_end(
        learner=learner,
        predictions=predictions,
        targets=targets,
        loss_dict={"loss": 0.5},
        stats=stats,
    )

    # Check if stats were collected
    # MetricsCallback appends "num_codes", "chance_probs", "chance_entropy", and "latent_root" (via add_latent_visualization)
    data = stats.get_data()
    assert "num_codes" in data
    assert "chance_probs" in data
    assert "chance_entropy" in data
    # Latent visualization is stored in latent_trackers if implemented in StatTracker
    # For now, asserting no error is thrown is a good smoke test.
