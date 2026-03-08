import pytest
import torch
from search.aos_search.scoring import ucb_score_fn, gumbel_score_fn
from search.aos_search.tree import FlatTree
from search.aos_search.min_max_stats import VectorizedMinMaxStats

pytestmark = pytest.mark.unit


def test_aos_ucb_score_fn():
    """Verifies fully vectorized UCB score calculations using the public function."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    tree = FlatTree.allocate(
        batch_size=1, max_nodes=2, num_actions=2, num_codes=1, device=device
    )

    tree.node_visits[0, 0] = 10
    tree.node_values[0, 0] = 0.5
    tree.children_visits[0, 0] = torch.tensor([5, 0], dtype=torch.int32)
    tree.children_values[0, 0] = torch.tensor([0.8, 0.0], dtype=torch.float32)
    tree.children_prior_logits[0, 0] = torch.log(
        torch.tensor([0.7, 0.3], dtype=torch.float32)
    )
    tree.children_action_mask[0, 0] = True

    min_max = VectorizedMinMaxStats.allocate(1, device)

    scores = ucb_score_fn(
        tree,
        torch.tensor([0], dtype=torch.int32),
        pb_c_init=1.25,
        pb_c_base=19652,
        min_max_stats=min_max,
    )

    assert scores.shape == (1, 2)
    assert not torch.isnan(scores).any()


def test_aos_gumbel_score_fn():
    """Verifies fully vectorized Gumbel score calculations using the public function."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    tree = FlatTree.allocate(
        batch_size=1, max_nodes=2, num_actions=2, num_codes=1, device=device
    )

    tree.node_visits[0, 0] = 10
    tree.children_visits[0, 0] = torch.tensor([3, 1], dtype=torch.int32)
    tree.children_values[0, 0] = torch.tensor([0.8, 0.0], dtype=torch.float32)
    tree.children_prior_logits[0, 0] = torch.log(
        torch.tensor([0.7, 0.3], dtype=torch.float32)
    )
    tree.children_action_mask[0, 0] = True

    min_max = VectorizedMinMaxStats.allocate(1, device)

    # FIXED: Argument names mapped to signature
    scores = gumbel_score_fn(
        tree,
        torch.tensor([0], dtype=torch.int32),
        gumbel_cvisit=50.0,
        gumbel_cscale=1.0,
        min_max_stats=min_max,
    )

    assert scores.shape == (1, 2)
    assert not torch.isnan(scores).any()
