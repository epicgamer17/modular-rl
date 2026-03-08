import pytest
import torch
from search.search_py.pruners import AlphaBetaPruning
from search.search_py.nodes import DecisionNode

pytestmark = pytest.mark.unit


class MockMinMaxStats:
    def normalize(self, val):
        return val


class MockChildNode:
    def __init__(self, value, visits=1):
        self.visits = visits
        self._value = value

    def expanded(self):
        return True

    def value(self):
        return self._value


def test_alpha_beta_pruning_cutoff():
    pruner = AlphaBetaPruning()

    # 1. Setup a node with a high-value child
    node = DecisionNode(prior=1.0)
    node.get_child_q_from_parent = lambda child: child.value()
    node.children = {0: MockChildNode(value=100.0)}  # A suspiciously good move

    # 2. Setup an artificially tight state where beta is very low
    # This simulates a scenario where the opponent would never let us reach this node
    state = {"alpha": 0.0, "beta": 50.0}

    # 3. Step the pruner
    # The current_best will evaluate to 100.0.
    # Alpha (0.0) updates to 100.0.
    # Because new Alpha (100.0) >= Beta (50.0), it MUST return an empty list (prune all).
    allowed_actions, next_state = pruner.step(
        node=node,
        state=state,
        config=None,
        min_max_stats=MockMinMaxStats(),
        current_sim_idx=0,
    )

    # Verify the branch was hit correctly
    assert allowed_actions == []
    assert next_state == state
