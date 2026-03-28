import math
import torch
import pytest
from search.aos_search.scoring import ucb_score_fn
from search.aos_search.tree import FlatTree

pytestmark = pytest.mark.unit

def test_puct_scoring_math_verification():
    """
    Verify the Upper Confidence Bound calculation using the MuZero PUCT formula.
    
    Setup:
    - Node visit count N(s) = 10
    - Action A: Q = 0.5, P = 0.3, N(s,a) = 5
    - Action B: Q = 0.2, P = 0.7, N(s,a) = 0
    
    Constants:
    - c1 (pb_c_init) = 1.25
    - c2 (pb_c_base) = 19652
    
    Expected math (hand-calculated):
    1. pb_c_multiplier = log((1 + 10 + 19652) / 19652) + 1.25
       log(19663 / 19652) ≈ 0.00055958285
       multiplier ≈ 1.25055958285
    
    Action A:
    - Exploration Factor = multiplier * sqrt(10) / (1 + 5)
      sqrt(10) ≈ 3.16227766
      factor ≈ 1.2505595828 * 3.16227766 / 6 ≈ 0.6591026
    - Prior Score = 0.6591026 * 0.3 ≈ 0.19773078
    - Normalized Q = (0.5 - 0.2) / (0.5 - 0.2) = 1.0
    - Total Score A ≈ 1.19773078
    
    Action B:
    - Exploration Factor = multiplier * sqrt(10) / (1 + 0)
      factor ≈ 1.2505595828 * 3.16227766 / 1 ≈ 3.954616
    - Prior Score = 3.954616 * 0.7 ≈ 2.7682312
    - Normalized Q = (0.2 - 0.2) / (0.5 - 0.2) = 0.0
    - Total Score B ≈ 2.7682312
    """
    device = torch.device("cpu")
    batch_size = 1
    max_nodes = 5
    num_actions = 2
    
    tree = FlatTree.allocate(batch_size, max_nodes, num_actions, 0, device)
    
    # We use node 0 as the parent
    node_idx = 0
    tree.node_visits[0, node_idx] = 10
    tree.node_values[0, node_idx] = 0.2 # parent value 0.2
    
    # Action A (Index 0)
    tree.children_visits[0, node_idx, 0] = 5
    tree.children_values[0, node_idx, 0] = 0.5
    tree.children_index[0, node_idx, 0] = 1 # Mark as expanded
    
    # Action B (Index 1)
    tree.children_visits[0, node_idx, 1] = 0
    tree.children_values[0, node_idx, 1] = 0.2
    tree.children_index[0, node_idx, 1] = 2 # Mark as expanded
    
    # Set logits to achieve priors [0.3, 0.7]
    # ln(0.7 / 0.3) ≈ 0.84729786
    tree.children_prior_logits[0, node_idx, 0] = 0.0
    tree.children_prior_logits[0, node_idx, 1] = 0.84729786
    
    # Ensure action mask is active
    tree.children_action_mask[0, node_idx, 0] = True
    tree.children_action_mask[0, node_idx, 1] = True
    
    # Constants
    pb_c_init = 1.25
    pb_c_base = 19652.0
    
    # Updated hand-calculated values (using more digits from the same formula)
    expected_score_a = 1.19773083
    expected_score_b = 2.76823163
    
    # Run the scoring function
    node_indices = torch.tensor([node_idx], dtype=torch.int32, device=device)
    scores = ucb_score_fn(
        tree,
        node_indices,
        pb_c_init=pb_c_init,
        pb_c_base=pb_c_base,
        bootstrap_method="parent_value"
    )
    
    score_a = scores[0, 0].item()
    score_b = scores[0, 1].item()
    
    # Assert specific math values
    assert math.isclose(score_a, expected_score_a, rel_tol=1e-7), f"Action A score {score_a} != {expected_score_a}"
    assert math.isclose(score_b, expected_score_b, rel_tol=1e-7), f"Action B score {score_b} != {expected_score_b}"
    
    # Assert selection
    assert score_b > score_a, f"Action B ({score_b}) should be selected over Action A ({score_a})"
