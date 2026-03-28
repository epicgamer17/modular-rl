import torch
import pytest
import math
from search.search_py.nodes import DecisionNode
from search.search_py.backpropogation import AverageDiscountedReturnBackpropagator
from search.search_py.min_max_stats import MinMaxStats

pytestmark = pytest.mark.unit

class MockConfig:
    def __init__(self, num_players, discount_factor=1.0):
        self.discount_factor = discount_factor
        self.game = type('obj', (object,), {'num_players': num_players})

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

def make_mock_search_path(path_config, num_players):
    """Reconstructs the Node tree based on the tuple config."""
    policy = torch.tensor([1.0, 0.0])
    
    root = DecisionNode(prior=1.0)
    root.expand(
        allowed_actions=torch.tensor([0, 1]),
        to_play=0,
        priors=policy,
        network_policy=policy,
        network_state={},
        reward=0.0,
        value=0.0
    )
    
    search_path = [root]
    action_path = []
    
    node = root
    for i, step in enumerate(path_config):
        to_play_next, reward = step[0], step[1]
        action = 0
        action_path.append(action)
        
        next_node = DecisionNode(prior=1.0, parent=node)
        node.children[action] = next_node
        
        next_node.expand(
            allowed_actions=torch.tensor([0, 1]),
            to_play=to_play_next,
            priors=policy,
            network_policy=policy,
            network_state={},
            reward=reward,
            value=0.0
        )
        
        node = next_node
        search_path.append(node)

    last = path_config[-1]
    leaf_value = last[2] if len(last) > 2 else 0.0
    leaf_to_play = last[0]
    
    return search_path, action_path, leaf_to_play, leaf_value

@pytest.mark.parametrize("path_config, expected, num_players, discount, desc", BACKPROP_CASES)
def test_muzero_multiplayer_backpropagation_py(path_config, expected, num_players, discount, desc):
    """Verifies the search_py backpropagation against analytical oracles (including discounting)."""
    search_path, action_path, leaf_to_play, leaf_value = make_mock_search_path(path_config, num_players)
    
    bp = AverageDiscountedReturnBackpropagator()
    min_max_stats = MinMaxStats(known_bounds=None)
    config = MockConfig(num_players=num_players, discount_factor=discount)
    
    bp.backpropagate(
        search_path=search_path,
        action_path=action_path,
        leaf_value=leaf_value,
        leaf_to_play=leaf_to_play,
        min_max_stats=min_max_stats,
        config=config
    )
    
    resulting_values = []
    for node in search_path:
        resulting_values.append(node.value())
        
    expected_tensor = torch.tensor(expected, dtype=torch.float32)
    resulting_tensor = torch.tensor(resulting_values, dtype=torch.float32)
    
    torch.testing.assert_close(resulting_tensor, expected_tensor, msg=f"Python Backend Failed: {desc}")
