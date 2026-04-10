import pytest
import torch
import numpy as np
from envs.factories.tictactoe import tictactoe_factory
from registries import (
    make_muzero_network,
    make_muzero_search_engine,
)
from utils.utils import action_mask_to_legal_moves

pytestmark = pytest.mark.integration

def test_muzero_tictactoe_masking():
    """
    Verifies that MuZero search never selects an illegal move in Tic-Tac-Toe.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Setup Env
    env = tictactoe_factory()
    env.reset()
    
    # 2. Pick a state with some illegal moves
    # Play a few moves: p1(0), p2(1), p1(2)
    # Positions 0, 1, 2 are now illegal.
    env.step(0)
    env.step(1)
    env.step(2)
    
    obs, reward, term, trunc, info = env.last()
    assert not term and not trunc
    
    # Verify current legal moves
    # info['legal_moves'] should be [3, 4, 5, 6, 7, 8]
    legal = info['legal_moves']
    print(f"Legal moves at test state: {legal}")
    assert 0 not in legal
    assert 1 not in legal
    assert 2 not in legal
    
    # 3. Setup MuZero
    obs_dim = (9, 3, 3)
    num_actions = 9
    
    network = make_muzero_network(
        obs_dim=obs_dim,
        num_actions=num_actions,
        device=torch.device("cpu")
    )
    
    search_engine = make_muzero_search_engine(
        num_simulations=50,  # Enough to explore
        num_actions=num_actions,
        device=torch.device("cpu")
    )
    
    # 4. Run Search
    # We want to see if it ever selects 0, 1, or 2.
    # We'll run it multiple times to be sure
    for i in range(10):
        # We need to wrap obs because Search expects a tensor
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        res = search_engine.run(
            obs_tensor,
            info,
            network,
            exploration=True
        )
        
        # Unpack modular_search results
        root_value, exploratory_policy, target_policy, best_action, search_metadata = res
        
        print(f"Trial {i}: Selected action {best_action}")
        
        # ASSERTION: Best action must be legal
        assert best_action in legal, f"Illegal action {best_action} selected! Legal: {legal}"
        
        # PROBS check: Probabilities (visit counts) for illegal actions must be 0
        # exploratory_policy is the visit distribution
        probs = exploratory_policy
        for illegal_idx in [0, 1, 2]:
            assert probs[illegal_idx] == 0, f"Found non-zero prob {probs[illegal_idx]} for illegal action {illegal_idx}"

    print("Success: MuZero masking integration test passed for Tic-Tac-Toe.")

if __name__ == "__main__":
    test_muzero_tictactoe_masking()
