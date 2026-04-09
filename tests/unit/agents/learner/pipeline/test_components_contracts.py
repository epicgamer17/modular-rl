import torch
import numpy as np
import pytest
from unittest.mock import MagicMock, call

from learner.pipeline.components import (
    MetricEarlyStopComponent,
    PriorityBufferUpdateComponent,
    BetaScheduleComponent,
    ResetNoiseComponent,
    EpsilonScheduleComponent,
    MPSCacheClearComponent,
)
from learner.core import Blackboard

pytestmark = pytest.mark.unit


def test_metric_early_stop_sets_flag_when_exceeded():
    """MetricEarlyStopComponent sets stop_execution when threshold is exceeded."""
    comp = MetricEarlyStopComponent(metric_key="approx_kl", threshold=0.01)

    bb = Blackboard()
    bb.losses = {"approx_kl": torch.tensor(0.02)}

    comp.execute(bb)
    assert bb.meta.get("stop_execution") is True


def test_metric_early_stop_no_flag_when_below():
    """MetricEarlyStopComponent does nothing when metric is below threshold."""
    comp = MetricEarlyStopComponent(metric_key="approx_kl", threshold=0.05)

    bb = Blackboard()
    bb.losses = {"approx_kl": torch.tensor(0.01)}

    comp.execute(bb)
    assert bb.meta.get("stop_execution") is None


def test_metric_early_stop_ignores_missing_key():
    """MetricEarlyStopComponent does nothing when metric key is absent."""
    comp = MetricEarlyStopComponent(metric_key="missing_key", threshold=0.01)

    bb = Blackboard()
    bb.losses = {"some_loss": torch.tensor(0.5)}

    comp.execute(bb)
    assert bb.meta.get("stop_execution") is None


def test_priority_buffer_update_calls_fn():
    """PriorityBufferUpdateComponent calls priority_update_fn with correct args."""
    mock_fn = MagicMock()
    comp = PriorityBufferUpdateComponent(priority_update_fn=mock_fn)

    indices = np.array([0, 1, 2])
    priorities = torch.tensor([0.5, 0.3, 0.1])

    bb = Blackboard(data={"indices": indices, "ids": None})
    bb.meta["priorities"] = priorities

    comp.execute(bb)

    mock_fn.assert_called_once()
    call_args = mock_fn.call_args
    np.testing.assert_array_equal(call_args[0][0], indices)
    np.testing.assert_array_almost_equal(call_args[0][1], priorities.numpy())
    assert call_args[1]["ids"] is None


def test_priority_buffer_update_noop_without_priorities():
    """PriorityBufferUpdateComponent does nothing when no priorities in meta."""
    mock_fn = MagicMock()
    comp = PriorityBufferUpdateComponent(priority_update_fn=mock_fn)

    bb = Blackboard(data={"indices": np.array([0])})

    comp.execute(bb)
    mock_fn.assert_not_called()


def test_beta_schedule_component_calls_set_beta():
    """BetaScheduleComponent calls set_beta_fn with schedule value."""
    mock_set_beta = MagicMock()
    mock_schedule = MagicMock()
    mock_schedule.get_value.return_value = 0.7

    comp = BetaScheduleComponent(
        set_beta_fn=mock_set_beta,
        per_beta_schedule=mock_schedule,
    )

    bb = Blackboard()
    comp.execute(bb)

    mock_schedule.get_value.assert_called_once()
    mock_set_beta.assert_called_once_with(0.7)


def test_reset_noise_component_calls_reset():
    """ResetNoiseComponent calls agent_network.reset_noise()."""
    mock_network = MagicMock()
    comp = ResetNoiseComponent(agent_network=mock_network)

    bb = Blackboard()
    comp.execute(bb)

    mock_network.reset_noise.assert_called_once()


def test_epsilon_schedule_component_steps():
    """EpsilonScheduleComponent calls epsilon_schedule.step()."""
    mock_schedule = MagicMock()
    comp = EpsilonScheduleComponent(epsilon_schedule=mock_schedule)

    bb = Blackboard()
    comp.execute(bb)

    mock_schedule.step.assert_called_once()


def test_mps_cache_clear_noop_on_cpu():
    """MPSCacheClearComponent does nothing on CPU device."""
    comp = MPSCacheClearComponent(device=torch.device("cpu"), interval=1)

    bb = Blackboard()
    comp.execute(bb)
    # No error = pass; MPS cache clear should not be called on CPU
