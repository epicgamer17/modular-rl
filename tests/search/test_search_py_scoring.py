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

pytestmark = pytest.mark.unit


class DummyMinMaxStats:
    def normalize(self, x):
        return x


class DummyChild:
    def __init__(self, expanded=True, val=0.0, prior=0.1, visits=1):
        self._expanded = expanded
        self._val = val
        self.prior = prior
        self.visits = visits

    def expanded(self):
        return self._expanded

    def value(self):
        return self._val


class DummyNode:
    def __init__(self):
        self.visits = 10
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.child_visits = torch.tensor([5.0, 0.0])
        self.child_priors = torch.tensor([0.7, 0.3])
        self.network_policy = torch.tensor([0.7, 0.3])
        self.child_values = torch.tensor([1.5, 0.0])
        self.children = {
            0: DummyChild(val=1.5),
            1: DummyChild(expanded=False, val=0.0, visits=0),
        }

    def value(self):
        return 0.5

    def get_v_mix(self):
        return 0.25

    def get_child_q_from_parent(self, child):
        return child.value() + 1.0

    def get_child_q_for_unvisited(self):
        return -1.0


def test_ucb_scoring_bootstraps():
    """Verifies UCB properly applies different bootstrap targets to unvisited nodes."""
    node = DummyNode()
    stats = DummyMinMaxStats()

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

    class DummyConfig:
        gumbel_cvisit = 50.0
        gumbel_cscale = 1.0

    scorer = GumbelScoring(DummyConfig())
    node = DummyNode()
    stats = DummyMinMaxStats()

    with pytest.raises(NotImplementedError):
        scorer.score(node, node.children[0], stats)

    scores = scorer.get_scores(node, stats)
    assert scores.shape == (2,)
    assert scorer.score_initial(0.5, 0) == 0.5


def test_least_visited_scoring():
    """Verifies basic negative visit sorting."""
    scorer = LeastVisitedScoring()
    node = DummyNode()
    stats = DummyMinMaxStats()

    assert scorer.score(node, node.children[0], stats) == -1.0
    scores = scorer.get_scores(node, stats)
    assert torch.allclose(scores, torch.tensor([-5.0, -0.0]))


def test_prior_scoring():
    """Verifies the raw policy prior is passed out unadjusted."""
    scorer = PriorScoring()
    node = DummyNode()
    stats = DummyMinMaxStats()

    assert scorer.score(node, node.children[0], stats) == 0.1
    assert scorer.score_initial(0.8, 0) == 0.8
    scores = scorer.get_scores(node, stats)
    assert torch.allclose(scores, torch.tensor([0.7, 0.3]))


def test_qvalue_scoring():
    """Verifies Q-value extraction correctly identifies expanded vs unexpanded states."""
    scorer = QValueScoring()
    node = DummyNode()
    stats = DummyMinMaxStats()

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
    node = DummyNode()
    stats = DummyMinMaxStats()

    with pytest.raises(NotImplementedError):
        scorer.score(node, node.children[0], stats)

    scores = scorer.get_scores(node, stats)
    # priors: [0.7, 0.3] / (visits + 1): [6.0, 1.0]
    assert torch.allclose(scores, torch.tensor([0.7 / 6.0, 0.3 / 1.0]))
    assert scorer.score_initial(0.9, 0) == 0.9
