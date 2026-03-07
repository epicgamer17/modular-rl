import pytest

pytestmark = pytest.mark.unit

import math
import torch
from search.nodes import DecisionNode, ChanceNode
import numpy as np


def make_search_path(path_config):
    """
    path_config is a list of tuples: (to_play, reward)
    Last element has the leaf value and to_play
    """
    root = DecisionNode(0.0)
    policy = torch.tensor([0.0, 1.0])
    hidden_state = torch.tensor([1])
    legal_moves = [0, 1]

    # Initialize root (player 0)
    root.expand(legal_moves, 0, policy, policy, hidden_state, 0.0)

    search_path = [root]
    node = root.get_child(0)

    # Build path according to config
    for i, (to_play, reward) in enumerate(path_config[:-1]):
        search_path.append(node)
        node.expand(legal_moves, to_play, policy, policy, hidden_state, reward)
        node = node.get_child(0)

    # Last node
    search_path.append(node)
    last_config = path_config[-1]
    leaf_to_play = last_config[0]
    leaf_reward = last_config[1]
    leaf_value = last_config[2] if len(last_config) > 2 else 0.0
    node.expand(legal_moves, leaf_to_play, policy, policy, hidden_state, leaf_reward)

    return search_path, leaf_to_play, leaf_value


def backpropagate_method_canonical(
    search_path, to_play, value, num_players, min_max_stats, discount=1.0
):
    """
    O(n) discounted backpropagation with correct sign handling for repeated same-player turns.
    Migrated from method 3 in the notebook.
    """
    n = len(search_path)
    if n == 0:
        return []

    acc = [0.0] * num_players
    for p in range(num_players):
        acc[p] = value if to_play == p else -value

    totals = [0.0] * n

    for i in range(n - 1, -1, -1):
        node = search_path[i]
        node_player = node.to_play
        totals[i] = acc[node_player]

        node.value_sum += totals[i]
        node.visits += 1

        if i > 0:
            r_i = node.reward  # Search path stores node reward which is incoming
            acting_player = search_path[i - 1].to_play
            for p in range(num_players):
                sign = 1.0 if acting_player == p else -1.0
                acc[p] = sign * r_i + discount * acc[p]

        # MinMaxStats update logic
        if i > 0:
            parent = search_path[i - 1]
            sign = 1.0 if (num_players == 1 or node.to_play == parent.to_play) else -1.0
        else:
            sign = 1.0

        parent_value_contrib = node.reward + discount * (sign * node.value())
        min_max_stats.update(parent_value_contrib)

    return [node.value() for node in search_path], min_max_stats


test_cases = [
    (
        [(1, 0.0), (1, 1.0), (0, 0.0, 0.0)],
        [-1.0, 1.0, 0.0, 0.0],
        2,
        "2-player: two player 1s with a reward for player 1 on a normally player 0 turn, ending on player 0",
    ),
    (
        [(1, 0.0), (1, 1.0), (1, 0.0, 0.0)],
        [-1.0, 1.0, 0.0, 0.0],
        2,
        "2-player: two player 1s with a reward for player 1 on a normally player 0 turn, ending on player 1",
    ),
    (
        [(1, 0.0), (1, 1.0), (0, 1.0, 0.0)],
        [-2.0, 2.0, 1.0, 0.0],
        2,
        "2-player: two player 1s both actions getting a reward (they should dont cancel), ending on a root for player 0",
    ),
    (
        [(1, 0.0), (1, 1.0), (1, 1.0, 0.0)],
        [-2.0, 2.0, 1.0, 0.0],
        2,
        "2-player: two player 1s both actions getting a reward (they should dont cancel), ending on a root for player 1",
    ),
    (
        [(1, 1.0), (1, 1.0), (0, 0.0, 0.0)],
        [0.0, 1.0, 0.0, 0.0],
        2,
        "2-player: Two player 1 turns (but player 0 got a reward), ending on player 0",
    ),
    (
        [(1, 1.0), (1, 1.0), (1, 0.0, 0.0)],
        [0.0, 1.0, 0.0, 0.0],
        2,
        "2-player: Two player 1 turns (but player 0 got a reward), ending on player 1",
    ),
    (
        [(1, 0.0), (0, 1.0), (1, 0.0, 0.0)],
        [-1.0, 1.0, 0.0, 0.0],
        2,
        "2-player: alternating game, player 1 wins on there first move",
    ),
    (
        [(1, 0.0), (0, 1.0), (1, 0.0), (0, 0.0, 0.0)],
        [-1.0, 1.0, 0.0, 0.0, 0.0],
        2,
        "2-player: alternating game, player 1 wins on there first move",
    ),
    (
        [(1, 0.0), (0, 0.0), (1, 1.0, 0.0)],
        [1.0, -1.0, 1.0, 0.0],
        2,
        "2-player: alternating game, player 0 wins",
    ),
    (
        [(1, 0.0), (0, 0.0), (1, 0.0), (0, 1.0, 0.0)],
        [-1.0, 1.0, -1.0, 1.0, 0.0],
        2,
        "2-player: alternating game, player 1 wins",
    ),
    (
        [(1, 0.0), (0, 0.0), (1, 0.0, 1.0)],
        [-1.0, 1.0, -1.0, 1.0],
        2,
        "2-player: alternating game with a leaf value",
    ),
    (
        [(1, 0.0), (0, 0.0), (1, 0.0), (0, 0.0, 1.0)],
        [1.0, -1.0, 1.0, -1.0, 1.0],
        2,
        "2-player: alternating game with a leaf value",
    ),
    (
        [(0, 1.0), (0, 1.0), (0, 1.0, 0.0)],
        [3.0, 2.0, 1.0, 0.0],
        2,
        "2-player: All player 0 turns",
    ),
    (
        [(0, 0.0), (0, 0.0), (0, 0.0, 4.0)],
        [4.0, 4.0, 4.0, 4.0],
        2,
        "2-player: All player 0 turns with leaf value",
    ),
    (
        [(1, 0.0), (1, 1.0), (0, 0.0, 4.0)],
        [3.0, -3.0, -4.0, 4.0],
        2,
        "2-player: Two player 1 turns with leaf value",
    ),
    (
        [(1, 0.0), (1, 1.0), (1, 0.0, 4.0)],
        [-5.0, 5.0, 4.0, 4.0],
        2,
        "2-player: Two player 1 turns with leaf value",
    ),
    (
        [(1, 0.0), (1, 1.0), (1, 0.0), (0, 0.0, 4.0)],
        [3.0, -3.0, -4.0, -4.0, 4.0],
        2,
        "2-player: Two player 1 turns with leaf value",
    ),
    # Single player test cases
    (
        [(0, 1.0), (0, 2.0), (0, 3.0, 0.0)],
        [6.0, 5.0, 3.0, 0.0],
        1,
        "1-player: All rewards sum up",
    ),
    (
        [(0, 1.0), (0, 0.0), (0, 0.0, 5.0)],
        [6.0, 5.0, 5.0, 5.0],
        1,
        "1-player: Rewards + leaf value",
    ),
    (
        [(1, 0.0), (2, 0.0), (0, 1.0), (0, 0.0, 0.0)],
        [-1.0, -1.0, 1.0, 0.0, 0.0],
        3,
        "3-player: Player 2 wins",
    ),
    (
        [(1, 1.0), (2, 0.0), (0, 0.0), (0, 0.0, 0.0)],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        3,
        "3-player: Player 0 wins",
    ),
    (
        [(1, 0.0), (2, 1.0), (0, 0.0), (0, 0.0, 0.0)],
        [-1.0, 1.0, 0.0, 0.0, 0.0],
        3,
        "3-player: Player 1 wins",
    ),
    (
        [(1, 0.0), (2, 0.0), (0, 0.0), (0, 1.0, 0.0)],
        [1.0, -1.0, -1.0, 1.0, 0.0],
        3,
        "3-player: Player 0 wins",
    ),
    (
        [(1, 0.0), (2, 0.0), (0, 0.0), (1, 0.0, 1.0)],
        [-1.0, 1.0, -1.0, -1.0, 1.0],
        3,
        "3-player: player 1 ends with a value prediction",
    ),
]


@pytest.mark.parametrize("path_config, expected, num_players, description", test_cases)
def test_muzero_backprop(path_config, expected, num_players, description):
    # Reset class attributes to ensure isolation from other tests
    DecisionNode.stochastic = False
    DecisionNode.discount = 1.0
    DecisionNode.bootstrap_method = "network_value"
    DecisionNode.use_value_prefix = False
    ChanceNode.discount = 1.0
    ChanceNode.bootstrap_method = "network_value"
    ChanceNode.use_value_prefix = False

    torch.manual_seed(42)
    np.random.seed(42)
    search_path, leaf_to_play, leaf_value = make_search_path(path_config)
    from search.min_max_stats import MinMaxStats

    min_max_stats = MinMaxStats(known_bounds=[-1, 1])

    results, _ = backpropagate_method_canonical(
        search_path, leaf_to_play, leaf_value, num_players, min_max_stats
    )

    for r, e in zip(results, expected):
        assert math.isclose(
            r, e, abs_tol=1e-5
        ), f"Failed {description}: expected {expected}, got {results}"


if __name__ == "__main__":
    pytest.main([__file__])
