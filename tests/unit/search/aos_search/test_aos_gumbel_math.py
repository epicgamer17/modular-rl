import pytest
import torch
import math
from search.aos_search.tree import FlatTree
from search.aos_search.scoring import compute_v_mix

# Test constants
GAMMA = 0.9
NET_POL = torch.tensor([0.6, 0.3, 0.1])
ROOT_VISITS = 4
ROOT_VALUE_SUM = 4.0 # root_v = 1.0

# Child 0: visits=10, value_sum=30.0, reward=1.0 -> q0 = 1.0 + 0.9 * 3.0 = 3.7
CHILD0_VISITS = 10
CHILD0_VALUE_SUM = 30.0
CHILD0_REWARD = 1.0

# Child 1: visits=5, value_sum=10.0, reward=0.5 -> q1 = 0.5 + 0.9 * 2.0 = 2.3
CHILD1_VISITS = 5
CHILD1_VALUE_SUM = 10.0
CHILD1_REWARD = 0.5

# Hand-calculated oracle
EXPECTED_V_MIX = 3.09375 # see search_py test for math

pytestmark = pytest.mark.unit

def test_gumbel_v_mix_math_verification():
    """
    Tests the vectorized compute_v_mix in the AOS backend using the FlatTree structure.
    """
    device = torch.device("cpu")
    batch_size = 1
    max_nodes = 5
    num_actions = 3
    
    # 1. Setup FlatTree
    tree = FlatTree.allocate(batch_size, max_nodes, num_actions, 0, device)
    
    node_idx = 0
    tree.node_visits[0, node_idx] = ROOT_VISITS
    tree.node_values[0, node_idx] = ROOT_VALUE_SUM / ROOT_VISITS
    tree.raw_network_values[0, node_idx] = 1.0 # Anchor to root_v
    
    # 2. Populate child stats
    logits = NET_POL.log()
    tree.children_prior_logits[0, node_idx] = logits
    tree.children_action_mask[0, node_idx] = True
    
    # Child 0
    tree.children_visits[0, node_idx, 0] = CHILD0_VISITS
    v0 = CHILD0_VALUE_SUM / CHILD0_VISITS
    q0 = CHILD0_REWARD + GAMMA * v0
    tree.children_values[0, node_idx, 0] = q0
    tree.children_index[0, node_idx, 0] = 1 # Mark as expanded (visited)
    
    # Child 1
    tree.children_visits[0, node_idx, 1] = CHILD1_VISITS
    v1 = CHILD1_VALUE_SUM / CHILD1_VISITS
    q1 = CHILD1_REWARD + GAMMA * v1
    tree.children_values[0, node_idx, 1] = q1
    tree.children_index[0, node_idx, 1] = 2 # Mark as expanded (visited)
    
    # Child 2: unvisited
    tree.children_visits[0, node_idx, 2] = 0
    tree.children_index[0, node_idx, 2] = -1 # Unvisited
    
    # 3. Calculate v_mix
    node_indices = torch.tensor([node_idx], dtype=torch.int32, device=device)
    vmix_tensor = compute_v_mix(tree, node_indices)
    vmix = vmix_tensor[0].item()
    
    # 4. Assert
    assert math.isclose(vmix, EXPECTED_V_MIX, rel_tol=1e-7), f"AOS v_mix {vmix} != {EXPECTED_V_MIX}"

def test_gumbel_v_mix_no_visits():
    """Edge case: v_mix returns raw network value when no children are visited."""
    device = torch.device("cpu")
    tree = FlatTree.allocate(1, 1, 3, 0, device)
    tree.raw_network_values[0, 0] = 0.5
    tree.children_index[0, 0] = -1 # All unvisited
    tree.children_action_mask[0, 0] = True
    # Initial logits produce uniform prior
    tree.children_prior_logits[0, 0] = torch.tensor([1e-12, 1e-12, 1e-12]).log()
    
    node_indices = torch.tensor([0], dtype=torch.int32, device=device)
    vmix_tensor = compute_v_mix(tree, node_indices)
    assert vmix_tensor[0].item() == 0.5
