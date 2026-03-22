import pytest
import math
import numpy as np
import torch

from search.nodes import DecisionNode
from search.search_py.backpropogation import AverageDiscountedReturnBackpropagator
from types import SimpleNamespace
from tests.search.conftest import DummyMinMaxStats

pytestmark = pytest.mark.unit


def make_search_path(path_config):
    """Builds a search path of DecisionNodes from a config list of (to_play, reward[, value])."""
    root = DecisionNode(prior=1.0)
    root.to_play = 0
    root.child_visits = torch.zeros(2)
    root.child_values = torch.zeros(2)

    search_path = [root]
    action_path = []

    current_node = root
    for i, (to_play, reward) in enumerate(path_config[:-1]):
        action_path.append(0)
        child = DecisionNode(parent=current_node, prior=1.0)
        child.to_play = to_play

        # FIXED: Assigning attribute explicitly rather than mocking the method
        child.reward = reward

        child.child_visits = torch.zeros(2)
        child.child_values = torch.zeros(2)

        search_path.append(child)
        current_node = child

    last_config = path_config[-1]
    leaf_to_play = last_config[0]
    leaf_reward = last_config[1]
    leaf_value = last_config[2] if len(last_config) > 2 else 0.0

    action_path.append(0)
    leaf = DecisionNode(parent=current_node, prior=1.0)
    leaf.to_play = leaf_to_play
    leaf.reward = leaf_reward
    search_path.append(leaf)

    return search_path, action_path, leaf_to_play, leaf_value


test_cases = [
    (
        [(1, 0.0), (1, 1.0), (0, 0.0, 0.0)],
        [-1.0, 1.0, 0.0, 0.0],
        2,
        "2-player: alternating ending p0",
    ),
    (
        [(1, 0.0), (1, 1.0), (1, 0.0, 0.0)],
        [-1.0, 1.0, 0.0, 0.0],
        2,
        "2-player: alternating ending p1",
    ),
    (
        [(0, 1.0), (0, 2.0), (0, 3.0, 0.0)],
        [6.0, 5.0, 3.0, 0.0],
        1,
        "1-player: All rewards sum up",
    ),
    (
        [(1, 0.0), (2, 0.0), (0, 1.0), (0, 0.0, 0.0)],
        [-1.0, -1.0, 1.0, 0.0, 0.0],
        3,
        "3-player: Player 2 wins",
    ),
]


@pytest.mark.parametrize("path_config, expected, num_players, description", test_cases)
def test_muzero_backprop_extended(path_config, expected, num_players, description):
    torch.manual_seed(42)
    np.random.seed(42)

    config = SimpleNamespace(
        game=SimpleNamespace(num_players=num_players), discount_factor=1.0
    )
    stats = DummyMinMaxStats()

    # Test directly against the canonical codebase propagator
    backprop = AverageDiscountedReturnBackpropagator()

    search_path, action_path, leaf_to_play, leaf_value = make_search_path(path_config)
    backprop.backpropagate(
        search_path, action_path, leaf_value, leaf_to_play, stats, config
    )

    for node, exp_val in zip(search_path, expected):
        # FIXED: Evaluate against node.value() so error logs print cleanly
        assert math.isclose(
            node.value(), exp_val, abs_tol=1e-5
        ), f"Failed {description}: Expected {exp_val}, got {node.value()}"
