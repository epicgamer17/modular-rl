import pytest
import torch
from search.search_py.search_selectors import SamplingSelection

# Canonical import path required for isinstance() to evaluate to True
from search.nodes import DecisionNode
from tests.search.conftest import DummyMinMaxStats, DummyScoringMethod, DummyChild

pytestmark = pytest.mark.unit


class MockDecisionNode(DecisionNode):
    """DecisionNode subclass with pre-configured children for sampling tests."""

    def __init__(self, priors):
        super().__init__(prior=1.0)
        self.to_play = 0
        self.child_priors = torch.tensor(priors)
        self.children = {i: DummyChild(idx=i) for i in range(len(priors))}

    def get_child(self, action):
        return self.children[action]


def test_sampling_selection_decision_node_temperature_0():
    """Verifies that T=0 collapses the sampling selection to a strict argmax."""
    scorer = DummyScoringMethod(torch.tensor([0.1, 0.8, 0.1]))
    selector = SamplingSelection(scoring_method=scorer, temperature=0.0)

    node = MockDecisionNode(priors=[1.0, 1.0, 1.0])

    action, child = selector.select_child(node, DummyMinMaxStats())
    assert action == 1
    assert child.idx == 1


def test_sampling_selection_decision_node_temperature_scaling():
    """Verifies temperature scaling correctly reshapes the probability distribution before multinomial sampling."""
    torch.manual_seed(42)
    scorer = DummyScoringMethod(torch.tensor([0.01, 0.98, 0.01]))

    selector = SamplingSelection(scoring_method=scorer, temperature=10.0)
    node = MockDecisionNode(priors=[1.0, 1.0, 1.0])

    sampled_actions = set()
    for _ in range(50):
        action, _ = selector.select_child(node, DummyMinMaxStats())
        sampled_actions.add(action)

    assert len(sampled_actions) > 1


def test_sampling_selection_illegal_masking():
    """Verifies that 0-prior actions are masked with -1e18, preventing them from being sampled."""
    torch.manual_seed(42)
    scorer = DummyScoringMethod(torch.tensor([0.5, 0.5, 0.5]))
    selector = SamplingSelection(scoring_method=scorer, temperature=1.0)

    node = MockDecisionNode(priors=[0.0, 1.0, 0.0])

    action, child = selector.select_child(node, DummyMinMaxStats())

    assert action == 1
