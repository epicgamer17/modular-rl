import torch
import math
import pytest
from search.aos_search.tree import FlatTree
from search.aos_search.batched_mcts import _backpropagate, UNEXPANDED_SENTINEL
from search.aos_search.backpropogation import average_discounted_backprop

pytestmark = pytest.mark.unit

def test_value_backpropagation_discounting():
    """
    Ensure values strictly discount and accumulate up the tree.
    
    Setup:
    - Root (Node 0) -> Action 0 -> Node 1 -> Action 0 -> Node 2
    - Reward transitioning from Node 1 to Node 2: 0.5
    - Leaf evaluation at Node 2: 1.0
    - Discount factor gamma: 0.9
    
    Expected Calculation:
    - Target Q at Node 1: r(Node 1, Node 2) + gamma * V(Node 2)
      = 0.5 + 0.9 * 1.0 = 1.4
    - Target Q at Node 0: r(Node 0, Node 1) + gamma * Q(Node 1)
      = 0.0 + 0.9 * 1.4 = 1.26
    """
    device = torch.device("cpu")
    batch_size = 1
    max_nodes = 5
    num_actions = 2
    
    # 1. Allocate tree
    tree = FlatTree.allocate(batch_size, max_nodes, num_actions, 0, device)
    
    # 2. Setup linear path relationships
    # Node 0 -> Action 0 -> Node 1 (already implicitly at index 1)
    # Node 1 -> Action 0 -> Node 2 (already implicitly at index 2)
    tree.children_index[0, 0, 0] = 1
    tree.children_index[0, 1, 0] = 2
    
    # 3. Setup rewards
    # Reward from Node 1 to Node 2 is 0.5
    tree.children_rewards[0, 1, 0] = 0.5
    
    # 4. Setup path data for backprop
    # Indices: [Root, Node 1, Node 2]
    path_nodes = torch.tensor([[0, 1, 2, 0, 0]], dtype=torch.int32, device=device)
    # Actions: [Action 0 from Root, Action 0 from Node 1]
    path_actions = torch.tensor([[0, 0, UNEXPANDED_SENTINEL, UNEXPANDED_SENTINEL]], dtype=torch.int32, device=device)
    # Depth is 2 (2 actions taken)
    depths = torch.tensor([2], dtype=torch.int32, device=device)
    
    # 5. Setup leaf value (V at node 2)
    leaf_values = torch.tensor([1.0], dtype=torch.float32, device=device)
    # Player ID (used for perspective)
    leaf_to_play = torch.tensor([0], dtype=torch.long, device=device)
    
    # 6. Run backpropagation
    discount = 0.9
    _backpropagate(
        tree=tree,
        path_nodes=path_nodes,
        path_actions=path_actions,
        depths=depths,
        leaf_values=leaf_values,
        leaf_to_play=leaf_to_play,
        discount=discount,
        B=batch_size,
        device=device,
        backprop_fn=average_discounted_backprop,
        num_players=1 # For simplicity, 1 player game
    )
    
    # 7. Assertions
    # Visit counts should increment by 1
    # Note: FlatTree.allocate might initialize them to 0. 
    # Root might have visits = 1 if it's the root simulation, but here we just check increment.
    assert tree.node_visits[0, 0] == 1, "Root visit count should be 1"
    assert tree.node_visits[0, 1] == 1, "Node 1 visit count should be 1"
    
    # Q-values (stored in children_values)
    # Node 1's Q-value for Action 0 (leading to Node 2)
    q_val_n1 = tree.children_values[0, 1, 0].item()
    expected_q_n1 = 1.4
    assert math.isclose(q_val_n1, expected_q_n1, rel_tol=1e-6), f"Node 1 Q-value {q_val_n1} != {expected_q_n1}"
    
    # Root's Q-value for Action 0 (leading to Node 1)
    q_val_root = tree.children_values[0, 0, 0].item()
    expected_q_root = 1.26
    assert math.isclose(q_val_root, expected_q_root, rel_tol=1e-6), f"Root Q-value {q_val_root} != {expected_q_root}"
    
    # Node values (V)
    # Node 1's value should match its mean Q
    assert math.isclose(tree.node_values[0, 1].item(), expected_q_n1, rel_tol=1e-6)
    assert math.isclose(tree.node_values[0, 0].item(), expected_q_root, rel_tol=1e-6)
