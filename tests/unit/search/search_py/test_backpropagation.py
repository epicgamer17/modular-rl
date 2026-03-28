import torch
import math
import pytest
from search.search_py.nodes import DecisionNode
from search.search_py.backpropogation import AverageDiscountedReturnBackpropagator
from search.search_py.min_max_stats import MinMaxStats

pytestmark = pytest.mark.unit

class MockConfig:
    def __init__(self, discount_factor=0.9):
        self.discount_factor = discount_factor
        self.game = type('obj', (object,), {'num_players': 1})

def test_search_py_value_backpropagation_discounting():
    """
    Ensure values strictly discount and accumulate up the tree in search_py.
    
    Setup:
    - Root (Node 0) -> Action 0 -> Node 1 -> Action 0 -> Node 2
    - Reward transitioning from Node 1 to Node 2: 0.5
    - Leaf evaluation at Node 2: 1.0 (Node 2's predicted value)
    - Discount factor gamma: 0.9
    
    Expected Calculation:
    - Target Q at Node 1: r(Node 1, Node 2) + gamma * V(Node 2)
      = 0.5 + 0.9 * 1.0 = 1.4
    - Target Q at Node 0: r(Node 0, Node 1) + gamma * Q(Node 1)
      = 0.0 + 0.9 * 1.4 = 1.26
    """
    # 1. Setup Nodes
    root = DecisionNode(prior=1.0)
    n1 = DecisionNode(prior=1.0, parent=root)
    n2 = DecisionNode(prior=1.0, parent=n1)
    
    # Expand nodes to initialize vectorized stats
    # num_actions = 1
    root.expand(
        allowed_actions=torch.tensor([0]),
        to_play=0,
        priors=torch.tensor([1.0]),
        network_policy=torch.tensor([1.0]),
        network_state={},
        reward=0.0,
        value=0.0
    )
    n1.expand(
        allowed_actions=torch.tensor([0]),
        to_play=0,
        priors=torch.tensor([1.0]),
        network_policy=torch.tensor([1.0]),
        network_state={},
        reward=0.0, # reward from Root -> n1
        value=0.0
    )
    n2.expand(
        allowed_actions=torch.tensor([0]),
        to_play=0,
        priors=torch.tensor([1.0]),
        network_policy=torch.tensor([1.0]),
        network_state={},
        reward=0.5, # reward from n1 -> n2
        value=1.0 # Leaf value
    )
    
    # 2. Setup Backprop
    bp = AverageDiscountedReturnBackpropagator()
    search_path = [root, n1, n2]
    action_path = [0, 0]
    leaf_value = 1.0
    leaf_to_play = 0
    min_max_stats = MinMaxStats(known_bounds=None)
    config = MockConfig(discount_factor=0.9)
    
    # 3. Run Backpropagation
    bp.backpropagate(
        search_path=search_path,
        action_path=action_path,
        leaf_value=leaf_value,
        leaf_to_play=leaf_to_play,
        min_max_stats=min_max_stats,
        config=config
    )
    
    # 4. Assertions
    # Visit counts
    assert root.visits == 1
    assert n1.visits == 1
    assert n2.visits == 1
    
    # Child stats
    # n1.child_values[0] should be 1.4
    val_n1 = n1.child_values[0].item()
    assert math.isclose(val_n1, 1.4, rel_tol=1e-6), f"n1 child val {val_n1} != 1.4"
    
    # root.child_values[0] should be 1.26
    val_root = root.child_values[0].item()
    assert math.isclose(val_root, 1.26, rel_tol=1e-6), f"root child val {val_root} != 1.26"
    
    # Node values (value_sum / visits)
    assert math.isclose(n1.value(), 1.4, rel_tol=1e-6), f"n1 node value {n1.value()} != 1.4"
    assert math.isclose(root.value(), 1.26, rel_tol=1e-6), f"root node value {root.value()} != 1.26"
