import numpy as np
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agents.learners.base import UniversalLearner
from agents.learners.callbacks import Callback, EarlyStopIteration
from agents.learners.target_builders import BaseTargetBuilder
from losses.losses import LossPipeline
from modules.world_models.inference_output import LearningOutput

pytestmark = pytest.mark.unit


def _minimal_config(**overrides):
    base = dict(clipnorm=0.0)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_universal_learner_init_sets_fields():
    config = _minimal_config()
    agent_network = MagicMock()
    learner = UniversalLearner(
        config=config,
        agent_network=agent_network,
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=None,
        loss_pipeline=None,
        optimizer=None,
        lr_scheduler=None,
        callbacks=None,
    )

    assert learner.config is config
    assert learner.agent_network is agent_network
    assert learner.training_step == 0


def test_universal_learner_step_calls_optimizer_and_callbacks():
    config = _minimal_config(clipnorm=1.0)

    param = torch.nn.Parameter(torch.randn(1, requires_grad=True))
    agent_network = MagicMock()
    agent_network.parameters.return_value = [param]

    target_builder = MagicMock(spec=BaseTargetBuilder)
    loss_pipeline = MagicMock(spec=LossPipeline)
    optimizer = MagicMock(spec=torch.optim.Optimizer)
    lr_scheduler = MagicMock()

    batch = {
        "observations": torch.randn(2, 4),
        "weights": torch.ones(2),
        "indices": np.array([0, 1]),
    }
    batch_iterator = [batch]

    predictions = LearningOutput(values=torch.randn(2, 1, requires_grad=True))
    agent_network.learner_inference.return_value = predictions

    targets = {"values": torch.randn(2, 1)}
    target_builder.build_targets.return_value = targets

    loss_val = torch.tensor(0.5, requires_grad=True)
    priorities = torch.tensor([0.1, 0.2])
    loss_pipeline.run.return_value = (loss_val, {"total_loss": 0.5}, priorities)

    callback = MagicMock(spec=Callback)

    learner = UniversalLearner(
        config=config,
        agent_network=agent_network,
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=target_builder,
        loss_pipeline=loss_pipeline,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        callbacks=[callback],
    )

    with patch("agents.learners.base.clip_grad_norm_") as mock_clip:
        stats = learner.step(batch_iterator=batch_iterator)

    optimizer.zero_grad.assert_called_once_with(set_to_none=True)
    optimizer.step.assert_called_once()
    lr_scheduler.step.assert_called_once()
    assert mock_clip.call_count == 1

    callback.on_step_begin.assert_called_once()
    callback.on_backward_end.assert_called_once()
    callback.on_optimizer_step_end.assert_called_once()
    callback.on_step_end.assert_called_once()
    callback.on_training_step_end.assert_called_once()

    assert learner.training_step == 1
    assert stats is not None
    assert stats["loss"] == pytest.approx(0.5)
    assert stats["total_loss"] == pytest.approx(0.5)


def test_universal_learner_early_stop_iteration_breaks_loop():
    config = _minimal_config(clipnorm=0.0)
    agent_network = MagicMock()
    agent_network.parameters.return_value = [
        torch.nn.Parameter(torch.randn(1, requires_grad=True))
    ]
    agent_network.learner_inference.return_value = LearningOutput(
        values=torch.randn(1, 1, requires_grad=True)
    )

    target_builder = MagicMock(spec=BaseTargetBuilder)
    target_builder.build_targets.return_value = {"values": torch.randn(1, 1)}

    loss_pipeline = MagicMock(spec=LossPipeline)
    loss_pipeline.run.return_value = (
        torch.tensor(0.5, requires_grad=True),
        {"l": 0.5},
        torch.zeros(1),
    )

    optimizer = MagicMock(spec=torch.optim.Optimizer)

    stop_cb = MagicMock(spec=Callback)
    stop_cb.on_backward_end.side_effect = EarlyStopIteration()

    learner = UniversalLearner(
        config=config,
        agent_network=agent_network,
        device=torch.device("cpu"),
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=target_builder,
        loss_pipeline=loss_pipeline,
        optimizer=optimizer,
        callbacks=[stop_cb],
    )

    batches = [{"observations": torch.randn(1, 4)} for _ in range(3)]
    learner.step(batch_iterator=batches)

    # Only the first batch should be processed before early stop
    assert optimizer.step.call_count == 1
    stop_cb.on_training_step_end.assert_called_once()


def test_universal_learner_save_load_checkpoint(tmp_path):
    config = _minimal_config()
    device = torch.device("cpu")

    net = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    learner = UniversalLearner(
        config=config,
        agent_network=net,
        device=device,
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=None,
        loss_pipeline=None,
        optimizer=opt,
    )
    learner.training_step = 123

    ckpt = tmp_path / "ckpt.pt"
    learner.save_checkpoint(str(ckpt))
    assert ckpt.exists()

    net2 = torch.nn.Linear(4, 2)
    opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    learner2 = UniversalLearner(
        config=config,
        agent_network=net2,
        device=device,
        num_actions=2,
        observation_dimensions=(4,),
        observation_dtype=torch.float32,
        target_builder=None,
        loss_pipeline=None,
        optimizer=opt2,
    )
    learner2.load_checkpoint(str(ckpt))

    assert learner2.training_step == 123
