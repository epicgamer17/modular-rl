import pytest
import math
from atomic_rl.schedules.exponential import get_exponential_schedule

pytestmark = pytest.mark.unit


def test_get_exponential_schedule():
    start, end = 1.0, 0.1
    decay_rate = 100.0

    # step 0: 0.1 + (1.0 - 0.1) * exp(0) = 0.1 + 0.9 = 1.0
    assert math.isclose(get_exponential_schedule(0, start, end, decay_rate), 1.0)
    # step 100: 0.1 + 0.9 * exp(-1) approx 0.1 + 0.9 * 0.367879 = 0.1 + 0.33109 = 0.43109
    expected = end + (start - end) * math.exp(-1.0)
    assert math.isclose(get_exponential_schedule(100, start, end, decay_rate), expected)


def test_exponential_schedule():
    """Test exponential schedule decay."""
    # start 1.0, end 0.1, decay_rate 10
    # val = end + (start - end) * exp(-step/rate)
    assert math.isclose(get_exponential_schedule(0, 1.0, 0.1, 10), 1.0)
    expected_middle = 0.1 + 0.9 * math.exp(-5 / 10)
    assert math.isclose(get_exponential_schedule(5, 1.0, 0.1, 10), expected_middle)
