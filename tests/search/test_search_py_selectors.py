import pytest
import torch
from search.search_py.search_selectors import TopScoreSelection, SamplingSelection

# FIXED: Using canonical absolute import path to preserve 'isinstance' validity
from search.nodes import DecisionNode, ChanceNode
from tests.search.conftest import DummyMinMaxStats, DummyScoringMethod

pytestmark = pytest.mark.unit


def _create_mock_decision_node(priors):
    node = DecisionNode(prior=1.0)
    node.to_play = 0
    node.child_priors = torch.tensor(priors)
    node.children = {
        i: DecisionNode(prior=priors[i], parent=node) for i in range(len(priors))
    }
    return node


def test_top_score_selection_tiebreak():
    primary_scorer = DummyScoringMethod(torch.tensor([1.0, 5.0, 5.0]))
    tiebreak_scorer = DummyScoringMethod(torch.tensor([0.0, 2.0, 10.0]))
    selector = TopScoreSelection(primary_scorer, tiebreak_scorer)
    node = _create_mock_decision_node(priors=[1.0, 1.0, 1.0])
    action, child = selector.select_child(node, DummyMinMaxStats())
    assert action == 2


def test_top_score_selection_illegal_masking_and_pruned_set():
    scorer = DummyScoringMethod(torch.tensor([10.0, 8.0, 6.0]))
    selector = TopScoreSelection(scorer)
    node = _create_mock_decision_node(priors=[0.0, 1.0, 1.0])
    action, _ = selector.select_child(node, DummyMinMaxStats())
    assert action == 1


def test_sampling_selection_chance_node():
    torch.manual_seed(42)
    selector = SamplingSelection()

    parent = DecisionNode(prior=1.0)
    parent.to_play = 0
    node = ChanceNode(prior=1.0, parent=parent)
    node.child_priors = torch.tensor([0.1, 0.8, 0.1])
    node.children = {
        0: DecisionNode(0.1, node),
        1: DecisionNode(0.8, node),
        2: DecisionNode(0.1, node),
    }

    code, child = selector.select_child(node, DummyMinMaxStats())
    assert code in [0, 1, 2]
    assert isinstance(child, DecisionNode)


def test_mask_actions_2d():
    selector = TopScoreSelection(DummyScoringMethod(torch.tensor([])))
    values = torch.zeros((2, 3))
    legal_moves = [[1, 2], [0]]
    masked = selector.mask_actions(values, legal_moves)
    assert masked[0, 0] == -float("inf")
    assert masked[1, 1] == -float("inf")
