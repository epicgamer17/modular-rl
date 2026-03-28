import math
import pytest

# Try to import the C++ backend via the search wrapper
try:
    import search
    search.set_backend("cpp")
    import search.search_cpp as search_cpp
except (ImportError, RuntimeError):
    pytest.skip("C++ search backend not available.", allow_module_level=True)

pytestmark = pytest.mark.unit

def test_search_cpp_value_backpropagation_discounting():
    """
    Ensure values strictly discount and accumulate up the tree in search_cpp.
    
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
    # 1. Setup Arena and Nodes
    arena = search_cpp.NodeArena()
    
    # Root
    root_idx = arena.create_decision(prior=1.0, parent_index=-1)
    root = arena.decision(root_idx)
    root.expand(to_play=0, network_policy=[1.0, 0.0], reward=0.0, network_value=0.0)
    
    # Node 1
    n1_idx = arena.create_decision(prior=1.0, parent_index=root_idx)
    n1 = arena.decision(n1_idx)
    n1.expand(to_play=0, network_policy=[1.0, 0.0], reward=0.0, network_value=0.0)
    root.set_child(0, n1_idx)
    
    # Node 2
    n2_idx = arena.create_decision(prior=1.0, parent_index=n1_idx)
    n2 = arena.decision(n2_idx)
    n2.expand(to_play=0, network_policy=[1.0, 0.0], reward=0.5, network_value=1.0)
    n1.set_child(0, n2_idx)
    
    # 2. Setup Backprop
    search_path = [root_idx, n1_idx, n2_idx]
    action_path = [0, 0]
    leaf_value = 1.0
    leaf_to_play = 0
    min_max_stats = search_cpp.MinMaxStats()
    
    config = search_cpp.BackpropConfig()
    config.discount_factor = 0.9
    config.num_players = 1
    
    # 3. Run Backpropagation
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
    
    # 4. Assertions
    # Visit counts
    # Backprop increments root, n1, and n2.
    assert arena.node(root_idx).visits == 1
    assert arena.node(n1_idx).visits == 1
    assert arena.node(n2_idx).visits == 1
    
    # Child stats
    # Root's child values
    root_node = arena.node(root_idx)
    # Action 0 child value should be 1.26
    assert math.isclose(root_node.child_values[0], 1.26, rel_tol=1e-6)
    
    # Node 1's child values
    n1_node = arena.node(n1_idx)
    # Action 0 child value should be 1.4
    assert math.isclose(n1_node.child_values[0], 1.4, rel_tol=1e-6)
    
    # Node values (V)
    assert math.isclose(arena.node(n1_idx).value(), 1.4, rel_tol=1e-6)
    assert math.isclose(arena.node(root_idx).value(), 1.26, rel_tol=1e-6)
