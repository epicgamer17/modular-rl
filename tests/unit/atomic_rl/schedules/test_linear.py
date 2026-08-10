import pytest
import math
from atomic_rl.schedules.linear import get_linear_schedule

pytestmark = pytest.mark.unit


def test_get_linear_schedule():
    start, end = 1.0, 0.1
    decay_steps = 100

    # Start
    assert math.isclose(get_linear_schedule(0, start, end, decay_steps), 1.0)
    # Middle (step 50)
    # 1.0 + 0.5 * (0.1 - 1.0) = 1.0 - 0.45 = 0.55
    assert math.isclose(get_linear_schedule(50, start, end, decay_steps), 0.55)
    # End
    assert math.isclose(get_linear_schedule(100, start, end, decay_steps), 0.1)
    # Capped
    assert math.isclose(get_linear_schedule(150, start, end, decay_steps), 0.1)


def test_get_linear_schedule_beta():
    # Testing beta-like use case (annealing up)
    start, end = 0.4, 1.0
    steps = 100
    assert math.isclose(get_linear_schedule(0, start, end, steps), 0.4)
    assert math.isclose(get_linear_schedule(50, start, end, steps), 0.7)
    assert math.isclose(get_linear_schedule(100, start, end, steps), 1.0)
    assert math.isclose(get_linear_schedule(150, start, end, steps), 1.0)


def test_linear_schedule():
    """Test linear schedule decay."""
    # start 1.0, end 0.1, decay_steps 10
    assert math.isclose(get_linear_schedule(0, 1.0, 0.1, 10), 1.0)
    assert math.isclose(
        get_linear_schedule(5, 1.0, 0.1, 10), 0.55
    )  # 1.0 + 0.5 * (-0.9)
    assert math.isclose(get_linear_schedule(10, 1.0, 0.1, 10), 0.1)
    assert math.isclose(
        get_linear_schedule(20, 1.0, 0.1, 10), 0.1
    )  # Capped at 1.0 fraction
