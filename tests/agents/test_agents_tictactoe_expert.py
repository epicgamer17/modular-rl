import pytest
import numpy as np
from agents.tictactoe_expert import TicTacToeBestAgent

pytestmark = pytest.mark.unit


def test_tictactoe_expert_predict():
    """Ensures base outputs are properly unmodified."""
    agent = TicTacToeBestAgent(name="test_expert")
    obs, info = agent.predict("dummy_obs", {"legal_moves": [0, 1]})
    assert obs == "dummy_obs"
    assert info == {"legal_moves": [0, 1]}
    assert agent.name == "test_expert"


def test_tictactoe_expert_select_actions_row_win():
    """Verifies agent takes the winning horizontal move."""
    np.random.seed(42)
    agent = TicTacToeBestAgent()

    # Board Representation: Plane 0 (Current Player), Plane 1 (Opponent)
    obs = np.zeros((2, 3, 3))
    obs[0, 0, 0] = 1
    obs[0, 0, 1] = 1  # Two pieces in the top row

    info = {"legal_moves": [2, 3, 4, 5, 6, 7, 8]}
    action = agent.select_actions((obs,), info)

    # Win in row 0, col 2 => index 2
    assert action == 2


def test_tictactoe_expert_select_actions_col_block():
    """Verifies agent prioritizes blocking an opponent's vertical win."""
    np.random.seed(42)
    agent = TicTacToeBestAgent()

    obs = np.zeros((2, 3, 3))
    obs[1, 0, 1] = 1
    obs[1, 1, 1] = 1  # Opponent has middle column setup

    info = {"legal_moves": [0, 2, 3, 5, 6, 7, 8]}
    action = agent.select_actions((obs,), info)

    # Block at row 2, col 1 => index 7
    assert action == 7


def test_tictactoe_expert_select_actions_diag_win():
    """Verifies diagonal win logic handles batched inputs correctly."""
    np.random.seed(42)
    agent = TicTacToeBestAgent()

    obs = np.zeros((2, 3, 3))
    obs[0, 0, 0] = 1
    obs[0, 1, 1] = 1

    info = {"legal_moves": [2, 3, 5, 6, 7, 8]}

    # Simulating standard batched observation (B, Planes, H, W)
    batched_obs = np.expand_dims(obs, axis=0)
    action = agent.select_actions(batched_obs, info)

    # Win at row 2, col 2 => index 8
    assert action == 8


def test_tictactoe_expert_select_actions_anti_diag_block():
    """Verifies anti-diagonal blocking capabilities."""
    np.random.seed(42)
    agent = TicTacToeBestAgent()

    obs = np.zeros((2, 3, 3))
    obs[1, 0, 2] = 1
    obs[1, 1, 1] = 1

    info = {"legal_moves": [0, 3, 5, 6, 7, 8]}
    action = agent.select_actions(obs, info)  # Passed as raw unbatched

    # Block at row 2, col 0 => index 6
    assert action == 6
