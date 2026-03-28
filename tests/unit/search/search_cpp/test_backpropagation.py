import math
import pytest
import torch

try:
    import search
    search.set_backend("cpp")
    import search.search_cpp as search_cpp
except (ImportError, RuntimeError):
    pytest.skip("C++ search backend not available.", allow_module_level=True)

pytestmark = pytest.mark.unit

# Analytical Oracles: (path_config, expected, num_players, discount, description)
BACKPROP_CASES = [
    ([(1, 0.0), (1, 1.0), (0, 0.0, 0.0)], [-1.0, 1.0, 0.0, 0.0], 2, 1.0, "2P: two p1s, reward for p1 on p0 turn, end on p0"),
    ([(1, 0.0), (1, 1.0), (1, 0.0, 0.0)], [-1.0, 1.0, 0.0, 0.0], 2, 1.0, "2P: two p1s, reward for p1 on p0 turn, end on p1"),
    ([(1, 0.0), (1, 1.0), (0, 1.0, 0.0)], [-2.0, 2.0, 1.0, 0.0], 2, 1.0, "2P: two p1s, both actions get reward, end on p0"),
    ([(1, 0.0), (1, 1.0), (1, 1.0, 0.0)], [-2.0, 2.0, 1.0, 0.0], 2, 1.0, "2P: two p1s, both actions get reward, end on p1"),
    ([(1, 1.0), (1, 1.0), (0, 0.0, 0.0)], [0.0, 1.0, 0.0, 0.0], 2, 1.0, "2P: Two p1 turns (but p0 got reward), end on p0"),
    ([(1, 1.0), (1, 1.0), (1, 0.0, 0.0)], [0.0, 1.0, 0.0, 0.0], 2, 1.0, "2P: Two p1 turns (but p0 got reward), end on p1"),
    ([(1, 0.0), (0, 1.0), (1, 0.0, 0.0)], [-1.0, 1.0, 0.0, 0.0], 2, 1.0, "2P: alt game, p1 wins on first move"),
    ([(1, 0.0), (0, 1.0), (1, 0.0), (0, 0.0, 0.0)], [-1.0, 1.0, 0.0, 0.0, 0.0], 2, 1.0, "2P: alt game, p1 wins on first move (extra leaf)"),
    ([(1, 0.0), (0, 0.0), (1, 1.0, 0.0)], [1.0, -1.0, 1.0, 0.0], 2, 1.0, "2P: alt game, p0 wins"),
    ([(1, 0.0), (0, 0.0), (1, 0.0), (0, 1.0, 0.0)], [-1.0, 1.0, -1.0, 1.0, 0.0], 2, 1.0, "2P: alt game, p1 wins"),
    ([(1, 0.0), (0, 0.0), (1, 0.0, 1.0)], [-1.0, 1.0, -1.0, 1.0], 2, 1.0, "2P: alt game with leaf value"),
    ([(1, 0.0), (0, 0.0), (1, 0.0), (0, 0.0, 1.0)], [1.0, -1.0, 1.0, -1.0, 1.0], 2, 1.0, "2P: alt game with leaf value"),
    ([(0, 1.0), (0, 1.0), (0, 1.0, 0.0)], [3.0, 2.0, 1.0, 0.0], 2, 1.0, "2P: All p0 turns"),
    ([(0, 0.0), (0, 0.0), (0, 0.0, 4.0)], [4.0, 4.0, 4.0, 4.0], 2, 1.0, "2P: All p0 turns with leaf value"),
    ([(1, 0.0), (1, 1.0), (0, 0.0, 4.0)], [3.0, -3.0, -4.0, 4.0], 2, 1.0, "2P: Two p1 turns with leaf value"),
    ([(1, 0.0), (1, 1.0), (1, 0.0, 4.0)], [-5.0, 5.0, 4.0, 4.0], 2, 1.0, "2P: Two p1 turns with leaf value"),
    ([(1, 0.0), (1, 1.0), (1, 0.0), (0, 0.0, 4.0)], [3.0, -3.0, -4.0, -4.0, 4.0], 2, 1.0, "2P: Two p1 turns with leaf value"),
    ([(0, 1.0), (0, 2.0), (0, 3.0, 0.0)], [6.0, 5.0, 3.0, 0.0], 1, 1.0, "1P: All rewards sum up"),
    ([(0, 1.0), (0, 0.0), (0, 0.0, 5.0)], [6.0, 5.0, 5.0, 5.0], 1, 1.0, "1P: Rewards + leaf value"),
    ([(1, 0.0), (2, 0.0), (0, 1.0), (0, 0.0, 0.0)], [-1.0, -1.0, 1.0, 0.0, 0.0], 3, 1.0, "3P: Player 2 wins"),
    
    # Discounting Cases (gamma < 1.0)
    ([(0, 1.0), (0, 1.0, 1.0)], [2.71, 1.9, 1.0], 1, 0.9, "1P: Discounting gamma=0.9"),
    ([(1, 1.0), (0, 0.0, 1.0)], [1.81, -0.9, 1.0], 2, 0.9, "2P: Discounting gamma=0.9, alt turns"),
    ([(1, 0.0), (1, 1.0, 1.0)], [-1.71, 1.9, 1.0], 2, 0.9, "2P: Discounting gamma=0.9, non-alt turns P1->P1"),
]

def make_mock_cpp_tree(path_config, num_players):
    """Reconstructs the NodeArena path based on the oracle configuration."""
    arena = search_cpp.NodeArena()
    policy = [1.0, 0.0]
    
    root_idx = arena.create_decision(prior=1.0, parent_index=-1)
    arena.decision(root_idx).expand(to_play=0, network_policy=policy, reward=0.0, network_value=0.0)
    
    search_path = [root_idx]
    action_path = []
    
    parent_idx = root_idx
    for i, step in enumerate(path_config):
        to_play_next, reward = step[0], step[1]
        action = 0
        action_path.append(action)
        
        child_idx = arena.create_decision(prior=1.0, parent_index=parent_idx)
        arena.decision(child_idx).expand(to_play=to_play_next, network_policy=policy, reward=reward, network_value=0.0)
        arena.decision(parent_idx).set_child(action, child_idx)
        
        search_path.append(child_idx)
        parent_idx = child_idx

    last = path_config[-1]
    leaf_value = last[2] if len(last) > 2 else 0.0
    leaf_to_play = last[0]
    
    return arena, search_path, action_path, leaf_to_play, leaf_value

@pytest.mark.parametrize("path_config, expected, num_players, discount, desc", BACKPROP_CASES)
def test_muzero_multiplayer_backpropagation_cpp(path_config, expected, num_players, discount, desc):
    """Verifies the search_cpp backpropagation against analytical oracles (including discounting)."""
    arena, search_path, action_path, leaf_to_play, leaf_value = make_mock_cpp_tree(path_config, num_players)
    
    min_max_stats = search_cpp.MinMaxStats()
    config = search_cpp.BackpropConfig()
    config.discount_factor = discount
    config.num_players = num_players
    
    search_cpp.backpropagate_with_method(
        search_cpp.BackpropMethodType.AVERAGE_DISCOUNTED_RETURN,
        arena,
        search_path,
        action_path,
        leaf_value,
        leaf_to_play,
        min_max_stats,
        config
    )
    
    resulting_values = []
    for node_idx in search_path:
        resulting_values.append(arena.node(node_idx).value())
        
    expected_tensor = torch.tensor(expected, dtype=torch.float32)
    resulting_tensor = torch.tensor(resulting_values, dtype=torch.float32)
    
    torch.testing.assert_close(resulting_tensor, expected_tensor, msg=f"CPP Backend Failed: {desc}")
