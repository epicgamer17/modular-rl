import torch
import pytest
import math
from search.aos_search.tree import FlatTree
from search.aos_search.batched_mcts import _backpropagate, UNEXPANDED_SENTINEL
from search.aos_search.backpropogation import average_discounted_backprop

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

def make_mock_tree(path_config, num_players, device):
    """Reconstructs the FlatTree path based on the oracle configuration."""
    B = 1
    max_nodes = len(path_config) + 1
    num_actions = 2
    tree = FlatTree.allocate(B, max_nodes, num_actions, 0, device)
    
    path_nodes_list = []
    path_actions_list = []
    
    current_node = 0
    tree.to_play[0, 0] = 0
    
    for i, step in enumerate(path_config):
        to_play_next, reward = step[0], step[1]
        path_nodes_list.append(current_node)
        
        next_node = i + 1
        action = 0 
        
        tree.children_index[0, current_node, action] = next_node
        tree.children_rewards[0, current_node, action] = reward
        tree.to_play[0, next_node] = to_play_next
        
        path_actions_list.append(action)
        current_node = next_node
        
    path_nodes_list.append(current_node)
    
    padded_nodes = torch.zeros((1, max_nodes), dtype=torch.int32, device=device)
    padded_nodes[0, :len(path_nodes_list)] = torch.tensor(path_nodes_list, dtype=torch.int32)
    
    padded_actions = torch.full((1, max_nodes - 1), UNEXPANDED_SENTINEL, dtype=torch.int32, device=device)
    padded_actions[0, :len(path_actions_list)] = torch.tensor(path_actions_list, dtype=torch.int32)
    
    depths = torch.tensor([len(path_actions_list)], dtype=torch.int32, device=device)
    
    last = path_config[-1]
    leaf_to_play = last[0]
    leaf_value = last[2] if len(last) > 2 else 0.0
    
    tree.node_values[0, current_node] = leaf_value
    tree.node_visits[0, current_node] = 1
    
    return tree, padded_nodes, padded_actions, depths, leaf_to_play, leaf_value

@pytest.mark.parametrize("path_config, expected, num_players, discount, desc", BACKPROP_CASES)
def test_muzero_multiplayer_backpropagation_aos(path_config, expected, num_players, discount, desc):
    """Verifies the aos_search backpropagation against analytical oracles (including discounting)."""
    device = torch.device("cpu")
    tree, path_nodes, path_actions, depths, leaf_to_play, leaf_value = make_mock_tree(path_config, num_players, device)
    
    _backpropagate(
        tree=tree,
        path_nodes=path_nodes,
        path_actions=path_actions,
        depths=depths,
        leaf_values=torch.tensor([leaf_value], dtype=torch.float32, device=device),
        leaf_to_play=torch.tensor([leaf_to_play], dtype=torch.long, device=device),
        discount=discount,
        B=1,
        device=device,
        backprop_fn=average_discounted_backprop,
        num_players=num_players
    )
    
    active_len = len(path_config) + 1
    resulting_values = []
    for i in range(active_len):
        resulting_values.append(tree.node_values[0, i].item())
        
    expected_tensor = torch.tensor(expected, dtype=torch.float32)
    resulting_tensor = torch.tensor(resulting_values, dtype=torch.float32)
    
    torch.testing.assert_close(resulting_tensor, expected_tensor, msg=f"AOS Backend Failed: {desc}")
