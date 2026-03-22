import pytest
import torch
from search.search_py.backpropogation import (
    AverageDiscountedReturnBackpropagator,
    MinimaxBackpropagator,
)
from search.search_py.nodes import DecisionNode
from tests.search.conftest import DummyMinMaxStats, DummyBackpropConfig

pytestmark = pytest.mark.unit


def test_average_discounted_return_backprop():
    config = DummyBackpropConfig()
    stats = DummyMinMaxStats()
    backprop = AverageDiscountedReturnBackpropagator()

    parent = DecisionNode(prior=1.0)
    parent.to_play = 0
    parent.child_visits = torch.zeros(2)
    parent.child_values = torch.zeros(2)
    parent.child_reward = lambda x: 1.0

    child = DecisionNode(parent=parent, prior=1.0)
    child.to_play = 1
    child.value_sum = 0.0
    child.visits = 0

    search_path = [parent, child]
    action_path = [1]

    backprop.backpropagate(
        search_path,
        action_path,
        leaf_value=10.0,
        leaf_to_play=1,
        min_max_stats=stats,
        config=config,
    )
    assert child.visits == 1
    assert child.value_sum == 10.0
    assert parent.child_visits[1] == 1


def test_minimax_backprop_state_updates():
    config = DummyBackpropConfig()
    stats = DummyMinMaxStats()
    backprop = MinimaxBackpropagator()

    parent = DecisionNode(prior=1.0)
    parent.to_play = 0
    parent.child_visits = torch.zeros(2)
    parent.child_values = torch.zeros(2)
    parent.get_child_q_from_parent = lambda x: x.value()

    child = DecisionNode(parent=parent, prior=1.0)
    child.to_play = 1
    child.value_sum = 0.0
    child.visits = 0
    child.value = lambda: 5.0

    search_path = [parent, child]
    action_path = [0]

    backprop.backpropagate(
        search_path,
        action_path,
        leaf_value=5.0,
        leaf_to_play=1,
        min_max_stats=stats,
        config=config,
    )
    assert parent.child_values[0] == 5.0
