import pytest
import math
from atomic_rl.schedules.ape_x_epsilon import get_ape_x_epsilon

pytestmark = pytest.mark.unit


def test_ape_x_epsilon():
    """Test Ape-X fixed epsilon calculation."""
    # If num_actors <= 1, return base_eps
    assert get_ape_x_epsilon(0, 1, base_eps=0.4) == 0.4

    # Check extremes for multiple actors
    # actor 0 should have base_eps ^ (1 + 0) = base_eps
    assert math.isclose(get_ape_x_epsilon(0, 5, base_eps=0.4), 0.4)
    # actor last should have base_eps ^ (1 + alpha)
    expected_last = 0.4 ** (1 + 7.0)
    assert math.isclose(get_ape_x_epsilon(4, 5, base_eps=0.4, alpha=7.0), expected_last)
