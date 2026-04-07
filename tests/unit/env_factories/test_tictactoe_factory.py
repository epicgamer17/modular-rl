import pytest
import numpy as np
from envs.factories import tictactoe_factory

pytestmark = pytest.mark.unit

def test_tictactoe_factory_shapes():
    """Verify that the factory returns an environment with correct observation shapes."""
    env = tictactoe_factory()
    env.reset()
    
    agent = env.possible_agents[0]
    obs_space = env.observation_space(agent)
    
    # Raw TicTacToe is (3, 3, 2). 
    # Frame stacking k=4 -> (3, 3, 8)
    # Player plane -> (3, 3, 9)
    # Channel Swap -> (9, 3, 3)
    
    assert obs_space.shape == (9, 3, 3), f"Expected shape (9, 3, 3), got {obs_space.shape}"
    
    obs, reward, term, trunc, info = env.last()
    assert obs.shape == (9, 3, 3)
    assert "legal_moves" in info
    assert "player" in info

def test_tictactoe_factory_stepping():
    """Verify that we can step through the environment."""
    env = tictactoe_factory()
    env.reset()
    
    for _ in range(5):
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            break
        
        legal_moves = info["legal_moves"]
        assert len(legal_moves) > 0
        
        action = legal_moves[0]
        env.step(action)
        
    env.close()
