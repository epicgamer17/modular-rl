import pytest
import torch
import torch.nn as nn

from utils.schedule import ScheduleConfig, create_schedule
from modules.utils import ScheduleLRScheduler

pytestmark = pytest.mark.unit


def _make_optimizer(lr=0.001):
    """Helper to create a simple optimizer for scheduler testing."""
    model = nn.Linear(4, 2)
    return torch.optim.Adam(model.parameters(), lr=lr)


def test_lr_scheduler_constant():
    """Tier 1: Verify constant LR scheduler behavior (Default)."""
    optimizer = _make_optimizer(lr=0.001)
    schedule_config = ScheduleConfig(type="constant", initial=0.001)
    schedule = create_schedule(schedule_config)
    scheduler = ScheduleLRScheduler(optimizer, schedule)

    # Constant scheduler should maintain the same LR after init
    assert scheduler.get_last_lr()[0] == pytest.approx(0.001)
    # Step and verify still constant
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.001)


def test_lr_scheduler_linear_default_decay():
    """Tier 1: Verify linear LR scheduler decays to the final value."""
    initial_lr = 0.01
    final_lr = 0.001
    decay_steps = 100

    optimizer = _make_optimizer(lr=initial_lr)
    schedule_config = ScheduleConfig(
        type="linear", initial=initial_lr, final=final_lr, decay_steps=decay_steps
    )
    schedule = create_schedule(schedule_config)
    scheduler = ScheduleLRScheduler(optimizer, schedule)

    # After all decay steps, should reach final
    for _ in range(decay_steps):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(final_lr, rel=1e-2)


def test_lr_scheduler_linear_annealing_to_zero():
    """Tier 1: [ANALYTICAL ORACLE] Verify linear LR annealing to 0.0."""
    initial_lr = 0.01
    final_lr = 0.0
    decay_steps = 2

    optimizer = _make_optimizer(lr=initial_lr)
    schedule_config = ScheduleConfig(
        type="linear", initial=initial_lr, final=final_lr, decay_steps=decay_steps
    )
    schedule = create_schedule(schedule_config)
    scheduler = ScheduleLRScheduler(optimizer, schedule)

    # Note: ScheduleLRScheduler.__init__ calls step() once internally (PyTorch LRScheduler behavior).
    # After construction, the internal schedule is already at step 1.
    # Step through the remaining decay and verify convergence to 0
    optimizer.step()
    scheduler.step()
    optimizer.step()
    scheduler.step()

    # After enough steps past decay_steps, LR should be at final (0.0)
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0, abs=1e-6)


def test_learner_integration_steps_scheduler():
    """Tier 1: Verify optimizer LR is updated when scheduler.step() is called."""
    initial_lr = 0.001
    final_lr = 0.0
    decay_steps = 10

    optimizer = _make_optimizer(lr=initial_lr)
    schedule_config = ScheduleConfig(
        type="linear", initial=initial_lr, final=final_lr, decay_steps=decay_steps
    )
    schedule = create_schedule(schedule_config)
    scheduler = ScheduleLRScheduler(optimizer, schedule)

    # Simulate what UniversalLearner.step does: optimizer.step() then scheduler.step()
    for _ in range(decay_steps):
        optimizer.step()
        scheduler.step()

    # After full decay, LR should be at or near 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-6)
