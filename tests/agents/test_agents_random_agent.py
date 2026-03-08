import pytest
import numpy as np
from agents.random_agent import RandomAgent

pytestmark = pytest.mark.unit


def test_random_agent_predict():
    """Verifies predict returns observation and info completely unmodified."""
    agent = RandomAgent(name="test_random")
    obs = np.array([1.0, 2.0, 3.0])
    info = {"legal_moves": [0, 1]}

    out_obs, out_info = agent.predict(obs, info)

    assert np.array_equal(out_obs, obs)
    assert out_info == info
    assert agent.name == "test_random"


def test_random_agent_select_actions_with_legal_moves():
    """Verifies that actions are restricted strictly to the legal moves mask."""
    np.random.seed(42)
    agent = RandomAgent()
    info = {"legal_moves": [1, 3, 5]}

    action = agent.select_actions(prediction=None, info=info)
    assert action in [1, 3, 5]


def test_random_agent_select_actions_fallback():
    """Verifies fallback to action '0' when no legal moves are provided."""
    agent = RandomAgent()
    info_empty = {"legal_moves": []}
    info_missing = {}

    action_empty = agent.select_actions(prediction=None, info=info_empty)
    action_missing = agent.select_actions(prediction=None, info=info_missing)

    assert action_empty == 0
    assert action_missing == 0
