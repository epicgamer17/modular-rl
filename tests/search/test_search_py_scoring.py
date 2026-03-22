import pytest
import torch
from search.search_py.scoring_methods import (
    UCBScoring,
    GumbelScoring,
    LeastVisitedScoring,
    PriorScoring,
    QValueScoring,
    DeterministicChanceScoring,
)
from tests.search.conftest import (
    DummyMinMaxStats,
    DummySearchConfig,
    make_dummy_scoring_node,
)

pytestmark = pytest.mark.unit


def test_ucb_scoring_bootstraps():
    """Verifies UCB properly applies different bootstrap targets to unvisited nodes."""
    node = make_dummy_scoring_node()
    stats = DummyMinMaxStats(normalize=True)

    # Test all string branches
    for method in ["parent_value", "zero", "v_mix", "mu_fpu", "other"]:
        scorer = UCBScoring(bootstrap_method=method)
        scores = scorer.get_scores(node, stats)
        assert scores.shape == (2,)

    scorer = UCBScoring()
    with pytest.raises(NotImplementedError):
        scorer.score(node, node.children[0], stats)


def test_gumbel_scoring():
    """Verifies Gumbel score extraction and abstract method crash."""
    scorer = GumbelScoring(DummySearchConfig())
    node = make_dummy_scoring_node()
    stats = DummyMinMaxStats(normalize=True)

    with pytest.raises(NotImplementedError):
        scorer.score(node, node.children[0], stats)

    scores = scorer.get_scores(node, stats)
    assert scores.shape == (2,)
    assert scorer.score_initial(0.5, 0) == 0.5


def test_least_visited_scoring():
    """Verifies basic negative visit sorting."""
    scorer = LeastVisitedScoring()
    node = make_dummy_scoring_node()
    stats = DummyMinMaxStats(normalize=True)

    assert scorer.score(node, node.children[0], stats) == -1.0
    scores = scorer.get_scores(node, stats)
    assert torch.allclose(scores, torch.tensor([-5.0, -0.0]))


def test_prior_scoring():
    """Verifies the raw policy prior is passed out unadjusted."""
    scorer = PriorScoring()
    node = make_dummy_scoring_node()
    stats = DummyMinMaxStats(normalize=True)

    assert scorer.score(node, node.children[0], stats) == 0.1
    assert scorer.score_initial(0.8, 0) == 0.8
    scores = scorer.get_scores(node, stats)
    assert torch.allclose(scores, torch.tensor([0.7, 0.3]))


def test_qvalue_scoring():
    """Verifies Q-value extraction correctly identifies expanded vs unexpanded states."""
    scorer = QValueScoring()
    node = make_dummy_scoring_node()
    stats = DummyMinMaxStats(normalize=True)

    # Child 0 is expanded -> get_child_q_from_parent
    assert scorer.score(node, node.children[0], stats) == 2.5
    # Child 1 is not expanded -> child.value()
    assert scorer.score(node, node.children[1], stats) == 0.0

    scores = scorer.get_scores(node, stats)
    assert torch.allclose(scores, torch.tensor([1.5, 0.0]))
    assert scorer.score_initial(0.4, 0) == 0.4


def test_deterministic_chance_scoring():
    """Verifies chance exploration ratios."""
    scorer = DeterministicChanceScoring()
    node = make_dummy_scoring_node()
    stats = DummyMinMaxStats(normalize=True)

    with pytest.raises(NotImplementedError):
        scorer.score(node, node.children[0], stats)

    scores = scorer.get_scores(node, stats)
    # priors: [0.7, 0.3] / (visits + 1): [6.0, 1.0]
    assert torch.allclose(scores, torch.tensor([0.7 / 6.0, 0.3 / 1.0]))
    assert scorer.score_initial(0.9, 0) == 0.9
