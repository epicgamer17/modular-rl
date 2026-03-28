import math
import torch
import pytest
from search.search_py.scoring_methods import UCBScoring
from search.search_py.nodes import DecisionNode
from search.search_py.min_max_stats import MinMaxStats

pytestmark = pytest.mark.unit

def test_search_py_puct_scoring_math_verification():
    """
    Verify the Upper Confidence Bound calculation using the MuZero PUCT formula
    for the search_py backend.
    
    Setup:
    - Node visit count N(s) = 10
    - Action A: Q = 0.5, P = 0.3, N(s,a) = 5
    - Action B: Q = 0.2, P = 0.7, N(s,a) = 0
    
    Expected math (hand-calculated):
    - expected_score_a = 1.19773083
    - expected_score_b = 2.76823163
    """
    node = DecisionNode(prior=1.0)
    node.visits = 10
    node.pb_c_init = 1.25
    node.pb_c_base = 19652.0
    
    # Setup children stats (DecisionNode uses tensors internally for vectorized scoring)
    num_actions = 2
    node.child_visits = torch.tensor([5.0, 0.0], dtype=torch.float32)
    node.child_values = torch.tensor([0.5, 0.2], dtype=torch.float32)
    node.child_priors = torch.tensor([0.3, 0.7], dtype=torch.float32)
    
    # Set parent value (used for bootstrap of unvisited). We want Q=0.2 for B.
    node.value_sum = 10 * 0.2 # node.value() will be 0.2
    
    min_max_stats = MinMaxStats(known_bounds=[0.2, 0.5])
    
    scoring = UCBScoring(bootstrap_method="parent_value")
    scores = scoring.get_scores(node, min_max_stats)
    
    score_a = scores[0].item()
    score_b = scores[1].item()
    
    expected_score_a = 1.19773083
    expected_score_b = 2.76823163
    
    assert math.isclose(score_a, expected_score_a, rel_tol=1e-7), f"Action A score {score_a} != {expected_score_a}"
    assert math.isclose(score_b, expected_score_b, rel_tol=1e-7), f"Action B score {score_b} != {expected_score_b}"
    assert score_b > score_a
