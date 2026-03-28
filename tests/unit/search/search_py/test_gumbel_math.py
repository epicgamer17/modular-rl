import pytest
import torch
import math
from search.search_py.nodes import DecisionNode

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

# Hand-calculated oracle:
# v0 = 30/10 = 3.0.  q0 = 1.0 + 0.9*3.0 = 3.7
# v1 = 10/5 = 2.0.   q1 = 0.5 + 0.9*2.0 = 2.3
# expected_q_vis = (0.6 * 3.7) + (0.3 * 2.3) = 2.22 + 0.69 = 2.91
# p_vis_sum = 0.6 + 0.3 = 0.9
# sum_N = 10 + 5 = 15.0
# term = 15.0 * (2.91 / 0.9) = 15.0 * 3.23333333 = 48.5
# root_v = 4.0 / 4 = 1.0
# expected_vmix = (1.0 + 48.5) / (1.0 + 15.0) = 49.5 / 16.0 = 3.09375
EXPECTED_V_MIX = 3.09375

pytestmark = pytest.mark.unit

def test_gumbel_v_mix_math_verification():
    """
    Directly tests the DecisionNode math in the Python OOP backend.
    """
    # 1. Setup root node
    root = DecisionNode(prior=1.0)
    root.visits = ROOT_VISITS
    root.value_sum = ROOT_VALUE_SUM
    root.network_value = 1.0
    root.discount = GAMMA
    
    # 2. Expand and mock child stats
    root.expand(
        allowed_actions=torch.tensor([0, 1, 2]),
        to_play=0,
        priors=NET_POL.clone(),
        network_policy=NET_POL.clone(),
        network_state={},
        reward=0.0,
        value=1.0
    )
    
    # Manually overwrite child stats to simulate 'visited' children
    root.child_visits[0] = CHILD0_VISITS
    root.child_visits[1] = CHILD1_VISITS
    root.child_visits[2] = 0 # unvisited
    
    # For DecisionNode, child_values stores Q-values: r + gamma * V(s')
    v0 = CHILD0_VALUE_SUM / CHILD0_VISITS
    v1 = CHILD1_VALUE_SUM / CHILD1_VISITS
    
    root.child_values[0] = CHILD0_REWARD + GAMMA * v0
    root.child_values[1] = CHILD1_REWARD + GAMMA * v1
    root.child_values[2] = 0.0
    
    # 3. Calculate v_mix
    vmix = root.get_v_mix()
    
    # 4. Assert
    assert math.isclose(vmix, EXPECTED_V_MIX, rel_tol=1e-7), f"Python v_mix {vmix} != {EXPECTED_V_MIX}"

def test_gumbel_v_mix_no_visits():
    """Edge case: v_mix returns raw network value when no children are visited."""
    root = DecisionNode(prior=1.0)
    root.visits = 1
    root.network_value = 0.5
    root.child_visits = torch.zeros(3)
    root.child_values = torch.zeros(3)
    root.child_priors = torch.tensor([0.3, 0.4, 0.3])
    
    assert root.get_v_mix() == 0.5
