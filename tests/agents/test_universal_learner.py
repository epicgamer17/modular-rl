import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import numpy as np
from typing import Dict, Any

from agents.learners.base import UniversalLearner, StepResult, Callback
from agents.learners.target_builders import BaseTargetBuilder
from losses.losses import LossPipeline
from modules.world_models.inference_output import LearningOutput

pytestmark = pytest.mark.unit


def test_universal_learner_init(muzero_config):
    """Test UniversalLearner initialization."""
    agent_network = MagicMock()
    target_builder = MagicMock(spec=BaseTargetBuilder)
    loss_pipeline = MagicMock(spec=LossPipeline)
    optimizer = MagicMock(spec=torch.optim.Optimizer)
    replay_buffer = MagicMock()

    learner = UniversalLearner(
        config=muzero_config,
        agent_network=agent_network,
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=target_builder,
        loss_pipeline=loss_pipeline,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
    )

    assert learner.config == muzero_config
    assert learner.agent_network == agent_network
    assert learner.target_builder == target_builder
    assert learner.loss_pipeline == loss_pipeline
    assert learner.optimizer == optimizer
    assert learner.replay_buffer == replay_buffer
    assert learner.training_step == 0


def test_universal_learner_early_exit(muzero_config):
    """Test that learner exits early if replay buffer is too small."""
    muzero_config.min_replay_buffer_size = 100
    replay_buffer = MagicMock()
    replay_buffer.size = 50

    learner = UniversalLearner(
        config=muzero_config,
        agent_network=MagicMock(),
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=MagicMock(),
        loss_pipeline=MagicMock(),
        optimizer=MagicMock(),
        replay_buffer=replay_buffer,
    )

    result = learner.step()
    assert result is None
    replay_buffer.sample.assert_not_called()


def test_universal_learner_full_step_cycle(muzero_config):
    """
    Test the complete step cycle of UniversalLearner.
    Verifies forward, backward, clipping, and optimizer steps.
    """
    torch.manual_seed(42)
    # Configure mock behavior
    muzero_config.min_replay_buffer_size = 0
    muzero_config.training_iterations = 1
    muzero_config.clipnorm = 1.0
    muzero_config.minibatch_size = 2

    agent_network = MagicMock()
    # Mock parameters for clip_grad_norm_
    param = torch.nn.Parameter(torch.randn(1, requires_grad=True))
    agent_network.parameters.return_value = [param]

    target_builder = MagicMock(spec=BaseTargetBuilder)
    loss_pipeline = MagicMock(spec=LossPipeline)
    optimizer = MagicMock(spec=torch.optim.Optimizer)
    replay_buffer = MagicMock()
    replay_buffer.size = 10

    batch = {
        "observations": torch.randn(2, 4),
        "indices": np.array([0, 1]),
        "weights": torch.ones(2),
    }
    replay_buffer.sample.return_value = batch

    predictions = LearningOutput(values=torch.randn(2, 1))
    agent_network.learner_inference.return_value = predictions


    targets = {"values": torch.randn(2, 1)}
    target_builder.build_targets.return_value = targets

    loss_val = torch.tensor(0.5, requires_grad=True)
    priorities = torch.tensor([0.1, 0.2])
    loss_pipeline.run.return_value = (loss_val, {"total_loss": 0.5}, priorities)

    learner = UniversalLearner(
        config=muzero_config,
        agent_network=agent_network,
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=target_builder,
        loss_pipeline=loss_pipeline,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
    )

    # Mock clip_grad_norm_ to ensure it's called
    with patch("agents.learners.base.clip_grad_norm_") as mock_clip:
        result = learner.step()

    # Verify sequence of calls
    optimizer.zero_grad.assert_called_once_with(set_to_none=True)
    agent_network.learner_inference.assert_called_once_with(batch)
    target_builder.build_targets.assert_called_once_with(
        batch, predictions, agent_network
    )
    loss_pipeline.run.assert_called_once()

    # Verify backward and clipping
    assert mock_clip.call_count == 1
    optimizer.step.assert_called_once()

    # Verify priority update
    replay_buffer.update_priorities.assert_called_once()
    args, kwargs = replay_buffer.update_priorities.call_args
    assert np.allclose(args[1], priorities.numpy())

    # Verify return value
    assert result["loss"] == 0.5
    assert result["total_loss"] == 0.5
    assert learner.training_step == 1


def test_universal_learner_multiple_iterations(muzero_config):
    """Verify that training_iterations correctly loops."""
    muzero_config.min_replay_buffer_size = 0
    muzero_config.training_iterations = 3

    agent_network = MagicMock()
    agent_network.learner_inference.return_value = LearningOutput(values=torch.randn(1, 1))
    target_builder = MagicMock(spec=BaseTargetBuilder)
    target_builder.build_targets.return_value = {"values": torch.randn(1, 1)}

    loss_pipeline = MagicMock(spec=LossPipeline)
    loss_pipeline.run.return_value = (torch.tensor(0.5, requires_grad=True), {"total_loss": 0.5}, None)
    optimizer = MagicMock(spec=torch.optim.Optimizer)
    replay_buffer = MagicMock()
    replay_buffer.size = 10
    replay_buffer.sample.return_value = {"observations": torch.randn(1, 4), "indices": np.array([0])}

    learner = UniversalLearner(
        config=muzero_config,
        agent_network=agent_network,
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=target_builder,
        loss_pipeline=loss_pipeline,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
    )

    learner.step()
    assert optimizer.step.call_count == 3


def test_universal_learner_clipnorm_zero(muzero_config):
    """Ensure clip_grad_norm_ is not called if clipnorm <= 0."""
    muzero_config.min_replay_buffer_size = 0
    muzero_config.training_iterations = 1
    muzero_config.clipnorm = 0

    agent_network = MagicMock()
    param = torch.nn.Parameter(torch.randn(1, requires_grad=True))
    agent_network.parameters.return_value = [param]
    agent_network.learner_inference.return_value = LearningOutput(values=torch.randn(1, 1))

    target_builder = MagicMock(spec=BaseTargetBuilder)
    target_builder.build_targets.return_value = {"values": torch.randn(1, 1)}
    
    replay_buffer = MagicMock()
    replay_buffer.size = 1
    replay_buffer.sample.return_value = {
        "observations": torch.randn(1, 4),
        "indices": np.array([0]),
        "weights": torch.ones(1),
    }

    loss_pipeline = MagicMock(spec=LossPipeline)
    loss_pipeline.run.return_value = (torch.tensor(0.5, requires_grad=True), {"total_loss": 0.5}, None)

    learner = UniversalLearner(
        config=muzero_config,
        agent_network=agent_network,
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=target_builder,
        loss_pipeline=loss_pipeline,
        optimizer=MagicMock(),
        replay_buffer=replay_buffer,
    )

    with patch("agents.learners.base.clip_grad_norm_") as mock_clip:
        learner.step()
        mock_clip.assert_not_called()


def test_universal_learner_lr_scheduler(muzero_config):
    """Test that lr_scheduler.step() is called."""
    muzero_config.min_replay_buffer_size = 0
    muzero_config.training_iterations = 1

    lr_scheduler = MagicMock()
    agent_network = MagicMock()
    agent_network.learner_inference.return_value = LearningOutput(values=torch.randn(1, 1))
    target_builder = MagicMock(spec=BaseTargetBuilder)
    target_builder.build_targets.return_value = {"values": torch.randn(1, 1)}

    replay_buffer = MagicMock()
    replay_buffer.size = 1
    replay_buffer.sample.return_value = {
        "observations": torch.randn(1, 4),
        "indices": np.array([0]),
        "weights": torch.ones(1),
    }

    loss_pipeline = MagicMock(spec=LossPipeline)
    loss_pipeline.run.return_value = (torch.tensor(0.5, requires_grad=True), {"total_loss": 0.5}, None)

    learner = UniversalLearner(
        config=muzero_config,
        agent_network=agent_network,
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=target_builder,
        loss_pipeline=loss_pipeline,
        optimizer=MagicMock(),
        replay_buffer=replay_buffer,
        lr_scheduler=lr_scheduler,
    )

    learner.step()
    lr_scheduler.step.assert_called_once()


def test_universal_learner_save_load_checkpoint(muzero_config, tmp_path):
    """Test checkpoint saving and loading."""
    agent_network = MagicMock()
    agent_network.state_dict.return_value = {"weights": torch.tensor([1.0])}

    optimizer = MagicMock(spec=torch.optim.Optimizer)
    optimizer.state_dict.return_value = {"opt": 1}

    learner = UniversalLearner(
        config=muzero_config,
        agent_network=agent_network,
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=MagicMock(),
        loss_pipeline=MagicMock(),
        optimizer=optimizer,
        replay_buffer=MagicMock(),
    )
    learner.training_step = 123

    ckpt_path = tmp_path / "ckpt.pt"
    learner.save_checkpoint(str(ckpt_path))
    assert ckpt_path.exists()

    # Load back
    learner2 = UniversalLearner(
        config=muzero_config,
        agent_network=MagicMock(),
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=MagicMock(),
        loss_pipeline=MagicMock(),
        optimizer=MagicMock(),
        replay_buffer=MagicMock(),
    )
    learner2.load_checkpoint(str(ckpt_path))

    learner2.agent_network.load_state_dict.assert_called_once()
    assert learner2.training_step == 123


def test_universal_learner_preprocess_observation(muzero_config):
    """Test observation preprocessing logic."""
    learner = UniversalLearner(
        config=muzero_config,
        agent_network=MagicMock(),
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=MagicMock(),
        loss_pipeline=MagicMock(),
        optimizer=MagicMock(),
        replay_buffer=MagicMock(),
    )

    # Test with numpy array
    obs_np = np.array([1, 2, 3, 4], dtype=np.float32)
    obs_torch = learner._preprocess_observation(obs_np)
    assert torch.is_tensor(obs_torch)
    assert obs_torch.shape == (1, 4)

    # Test with tensor already on device
    obs_in = torch.randn(4)
    obs_out = learner._preprocess_observation(obs_in)
    assert obs_out.shape == (1, 4)


def test_universal_learner_stats_and_hooks(muzero_config):
    learner = UniversalLearner(
        config=muzero_config,
        agent_network=MagicMock(),
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=MagicMock(),
        loss_pipeline=MagicMock(),
        optimizer=MagicMock(),
        replay_buffer=MagicMock(),
    )

    stats = learner._prepare_stats({"loss_a": 0.1}, 0.2)
    assert stats["loss_a"] == 0.1
    assert stats["loss"] == 0.2

    # after_optimizer_step should not crash
    learner.after_optimizer_step({}, MagicMock(), None)


def test_universal_learner_callbacks(muzero_config):
    config = MagicMock()
    config.min_replay_buffer_size = 0
    config.training_iterations = 1
    config.clipnorm = 0
    config.minibatch_size = 1
    config.per_beta_schedule = None
    config.epsilon_schedule = None

    callback = MagicMock(spec=Callback)
    replay_buffer = MagicMock()
    replay_buffer.size = 1
    replay_buffer.sample.return_value = {
        "observations": torch.randn(1, 4),
        "indices": np.array([0]),
    }
    # Wait, simple batch is still a dict, only network/target outputs are LearningOutput

    loss_pipeline = MagicMock()
    loss_pipeline.run.return_value = (
        torch.tensor(0.5, requires_grad=True),
        {"l": 0.5},
        torch.zeros(1),
    )

    agent_network = MagicMock()
    agent_network.learner_inference.return_value = LearningOutput(values=torch.randn(1, 1))
    target_builder = MagicMock(spec=BaseTargetBuilder)
    target_builder.build_targets.return_value = {"values": torch.randn(1, 1)}

    learner = UniversalLearner(
        config=config,
        agent_network=agent_network,
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=target_builder,
        loss_pipeline=loss_pipeline,
        optimizer=MagicMock(),
        replay_buffer=replay_buffer,
        callbacks=[callback],
    )

    learner.step()
    callback.on_step_end.assert_called_once()
