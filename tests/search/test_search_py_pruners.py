import pytest
from search.search_py.pruners import NoPruning, AlphaBetaPruning

pytestmark = pytest.mark.unit


# --- Pure Python Dummies ---
class DummyChild:
    def __init__(self, is_expanded=True, visits=1):
        self._expanded = is_expanded
        self.visits = visits

    def expanded(self):
        return self._expanded


class DummyNode:
    def __init__(self, child_q_values):
        self.children = {}
        self.child_q_values = child_q_values  # Mock lookup
        for action, q in child_q_values.items():
            self.children[action] = DummyChild()

    def get_child_q_from_parent(self, child):
        # Reverse lookup for simplicity
        for act, ch in self.children.items():
            if ch is child:
                return self.child_q_values[act]
        return 0.0


def test_no_pruning_behavior():
    """Verifies NoPruning allows all actions silently."""
    pruner = NoPruning()
    init_state = pruner.initialize(root=None, config=None)
    assert init_state is None

    actions, next_state = pruner.step(None, None, None, None, 0)
    assert actions is None
    assert next_state is None
    assert pruner.mask_target_policy is False


def test_alpha_beta_pruning_cutoff():
    """Verifies AlphaBetaPruning triggers a beta cutoff when child Q > beta."""
    pruner = AlphaBetaPruning()
    state = pruner.initialize(root=None, config=None)

    assert state == {"alpha": -float("inf"), "beta": float("inf")}
    assert pruner.mask_target_policy is False

    # Inject a state where Beta is low, and we find a child with a higher Q value
    # This simulates a branch that is "too good" (opponent wouldn't allow it)
    state["beta"] = 5.0

    node = DummyNode({0: 2.0, 1: 10.0})  # Action 1 has Q=10.0, which > Beta=5.0

    actions, next_state = pruner.step(
        node, state, config=None, min_max_stats=None, current_sim_idx=0
    )

    # When alpha >= beta, it returns an empty list to indicate ALL actions are pruned.
    assert actions == []
    # State is preserved/passed through on cutoff
    assert next_state == state


def test_alpha_beta_pruning_negamax_flip():
    """Verifies the negamax flip for the next depth layer."""
    pruner = AlphaBetaPruning()
    state = {"alpha": 2.0, "beta": 10.0}

    node = DummyNode(
        {0: 3.0}
    )  # Child Q = 3.0, bumps alpha to 3.0. Does not exceed beta(10.0).

    actions, next_state = pruner.step(
        node, state, config=None, min_max_stats=None, current_sim_idx=0
    )

    # None means "allow all"
    assert actions is None
    # Next layer should flip and negate bounds: next_alpha = -beta, next_beta = -alpha
    assert next_state == {"alpha": -10.0, "beta": -3.0}
