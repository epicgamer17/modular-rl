import pytest
import torch
from search.aos_search.backpropogation import (
    average_discounted_backprop,
    minimax_backprop,
)
from tests.search.conftest import MockFlatTree

pytestmark = pytest.mark.unit


def test_aos_average_discounted_backprop():
    """Verifies vectorized incremental mean updates for batched node/child structures."""
    torch.manual_seed(42)
    tree = MockFlatTree(batch_size=2, num_nodes=5, num_edges=4)

    batch_idx = torch.arange(2)
    nodes_at_d = torch.tensor([1, 2], dtype=torch.int32)  # Parents
    actions_at_d = torch.tensor([0, 3], dtype=torch.int32)  # Actions taken
    current_values = torch.tensor([10.0, 5.0], dtype=torch.float32)
    valid_mask = torch.tensor([True, True], dtype=torch.bool)

    # First update (Visits go 0 -> 1)
    average_discounted_backprop(
        tree, batch_idx, nodes_at_d, actions_at_d, current_values, 1.0, valid_mask
    )

    assert tree.node_visits[0, 1] == 1
    assert tree.node_values[0, 1] == 10.0
    assert tree.children_visits[1, 2, 3] == 1
    assert tree.children_values[1, 2, 3] == 5.0

    # Second update to Batch 0, Node 1 with new value (Incremental mean: (10 + 20)/2 = 15)
    new_values = torch.tensor([20.0, 0.0], dtype=torch.float32)
    valid_mask_2 = torch.tensor([True, False], dtype=torch.bool)  # Ignore batch 1

    average_discounted_backprop(
        tree, batch_idx, nodes_at_d, actions_at_d, new_values, 1.0, valid_mask_2
    )

    assert tree.node_visits[0, 1] == 2
    assert tree.node_values[0, 1] == 15.0  # Correct mean computation

    # Batch 1 should be completely untouched due to valid_mask
    assert tree.node_visits[1, 2] == 1
    assert tree.node_values[1, 2] == 5.0


def test_aos_minimax_backprop():
    """Verifies vectorized minimax properly calculates max sibling Q-values."""
    torch.manual_seed(42)
    tree = MockFlatTree(batch_size=1, num_nodes=2, num_edges=3)

    batch_idx = torch.arange(1)
    nodes_at_d = torch.tensor([0], dtype=torch.int32)
    actions_at_d = torch.tensor([1], dtype=torch.int32)

    # Simulate siblings having already been visited and valued
    tree.children_visits[0, 0, 0] = 5
    tree.children_values[0, 0, 0] = 7.0

    # Current action path evaluation
    current_values = torch.tensor([15.0], dtype=torch.float32)
    valid_mask = torch.tensor([True], dtype=torch.bool)

    best_q = minimax_backprop(
        tree, batch_idx, nodes_at_d, actions_at_d, current_values, 1.0, valid_mask
    )

    # The action taken scored 15.0. Sibling 0 scored 7.0.
    # Max sibling over valid visited edges should be 15.0.
    assert best_q[0] == 15.0
    assert tree.node_values[0, 0] == 15.0
