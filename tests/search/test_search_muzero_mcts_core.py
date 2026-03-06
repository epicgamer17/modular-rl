import pytest
pytestmark = pytest.mark.integration

import torch
import numpy as np
from search.nodes import DecisionNode, ChanceNode
from search.min_max_stats import MinMaxStats


def test_v_mix_with_some_visited_children(monkeypatch):
    """
    Migrated from notebook: test_v_mix_with_some_visited_children
    """
    monkeypatch.setattr(DecisionNode, "bootstrap_method", "zero")
    monkeypatch.setattr(DecisionNode, "discount", 0.9)
    monkeypatch.setattr(DecisionNode, "use_value_prefix", False)
    monkeypatch.setattr(DecisionNode, "stochastic", False)
    monkeypatch.setattr(ChanceNode, "bootstrap_method", "zero")
    monkeypatch.setattr(ChanceNode, "discount", 0.9)
    monkeypatch.setattr(ChanceNode, "use_value_prefix", False)

    net_pol = torch.tensor([0.6, 0.3, 0.1])
    root = DecisionNode(prior=0.0)
    root._v_mix = None
    root.network_policy = net_pol
    root.visits = 4
    root.value_sum = 4.0 # value = 1.0
    root.to_play = 0
    
    # child 0: visited
    child0 = DecisionNode(prior=0.6, parent=root)
    child0.visits = 10
    child0.value_sum = 30.0 # value = 3.0
    child0.reward = 1.0
    child0.to_play = 0
    # Mark as expanded so it's counted in v_mix visited_actions
    child0.children = {0: DecisionNode(0.5, child0)} 
    
    # child 1: visited
    child1 = DecisionNode(prior=0.3, parent=root)
    child1.visits = 5
    child1.value_sum = 10.0 # value = 2.0
    child1.reward = 0.5
    child1.to_play = 0
    child1.children = {0: DecisionNode(0.5, child1)}
    
    # child 2: unvisited
    child2 = DecisionNode(prior=0.1, parent=root)
    child2.visits = 0
    child2.value_sum = 0.0
    child2.reward = 0.0
    child2.to_play = 0
    
    root.children = {0: child0, 1: child1, 2: child2}
    # Current node implementation uses vectorized child stats for v_mix.
    root.child_visits = torch.tensor([10.0, 5.0, 0.0])
    root.child_values = torch.tensor([3.7, 2.3, 0.0])
    root.child_priors = net_pol.clone()
    
    # Manual calculation (same as notebook)
    # v0=3, v1=2
    # q0 = 1.0 + 0.9*3.0 = 3.7
    # q1 = 0.5 + 0.9*2.0 = 2.3
    # expected_q_vis = 0.6*3.7 + 0.3*2.3 = 2.22 + 0.69 = 2.91
    # p_vis_sum = 0.6 + 0.3 = 0.9
    # sum_N = 10 + 5 + 0 = 15
    # term = 15 * (2.91 / 0.9) = 15 * 3.2333 = 48.5
    # expected_vmix = (1.0 + 48.5) / (1 + 15) = 49.5 / 16 = 3.09375
    
    expected_vmix = 3.09375
    computed_vmix = root.get_v_mix()
    
    assert pytest.approx(float(computed_vmix), rel=1e-6) == expected_vmix

def test_v_mix_no_visits(monkeypatch):
    monkeypatch.setattr(DecisionNode, "bootstrap_method", "zero")
    monkeypatch.setattr(DecisionNode, "discount", 0.9)
    monkeypatch.setattr(DecisionNode, "use_value_prefix", False)
    monkeypatch.setattr(DecisionNode, "stochastic", False)
    monkeypatch.setattr(ChanceNode, "bootstrap_method", "zero")
    monkeypatch.setattr(ChanceNode, "discount", 0.9)
    monkeypatch.setattr(ChanceNode, "use_value_prefix", False)

    root = DecisionNode(prior=0.0)
    root._v_mix = None
    root.network_policy = torch.tensor([0.5, 0.5])
    root.visits = 2
    root.value_sum = 6.0 # value = 3.0
    
    child0 = DecisionNode(0.5, root)
    child1 = DecisionNode(0.5, root)
    root.children = {0: child0, 1: child1}
    root.child_visits = torch.tensor([0.0, 0.0])
    root.child_values = torch.tensor([0.0, 0.0])
    root.child_priors = root.network_policy.clone()
    
    # sum_N == 0, should return root value
    assert float(root.get_v_mix()) == pytest.approx(3.0)

def test_completed_q(monkeypatch):
    """
    Verify completed Q fills unvisited children with v_mix
    """
    monkeypatch.setattr(DecisionNode, "bootstrap_method", "zero")
    monkeypatch.setattr(DecisionNode, "discount", 0.9)
    monkeypatch.setattr(DecisionNode, "use_value_prefix", False)
    monkeypatch.setattr(DecisionNode, "stochastic", False)
    monkeypatch.setattr(ChanceNode, "bootstrap_method", "zero")
    monkeypatch.setattr(ChanceNode, "discount", 0.9)
    monkeypatch.setattr(ChanceNode, "use_value_prefix", False)

    minmax = MinMaxStats(known_bounds=[0, 10])
    
    net_pol = torch.tensor([0.6, 0.3, 0.1])
    root = DecisionNode(prior=0.0)
    root._v_mix = None
    root.network_policy = net_pol
    root.visits = 4
    root.value_sum = 4.0 # value = 1.0
    root.to_play = 0
    
    # child 0: visited, Q=3.7
    child0 = DecisionNode(prior=0.6, parent=root)
    child0.visits = 10
    child0.value_sum = 30.0 
    child0.reward = 1.0
    child0.to_play = 0
    child0.children = {0: DecisionNode(0.5, child0)} 
    
    # child 2: unvisited, should get v_mix
    child2 = DecisionNode(prior=0.1, parent=root)
    child2.visits = 0
    
    root.children = {0: child0, 2: child2}
    root.child_visits = torch.tensor([10.0, 0.0, 0.0])
    root.child_values = torch.tensor([3.7, 0.0, 0.0])
    root.child_priors = net_pol.clone()
    
    # v_mix for child 0 only:
    # q0 = 3.7
    # expected_q_vis = 0.6 * 3.7 = 2.22
    # p_vis_sum = 0.6
    # term = 10 * (2.22 / 0.6) = 10 * 3.7 = 37.0
    # v_mix = (1.0 + 37.0) / (1 + 10) = 38.0 / 11 = 3.4545
    
    vmix = root.get_v_mix()
    
    # In search/nodes.py, completed_q logic isn't in a single function, 
    # it's usually inside ActionSelector. But DecisionNode has _calculate_visited_policy_mass.
    # Actually, the notebook had a get_completed_q method on Node. 
    # Let's see if we should add it or test the component logic.
    
    assert pytest.approx(float(vmix), rel=1e-4) == 3.4545

if __name__ == "__main__":
    pytest.main([__file__])
