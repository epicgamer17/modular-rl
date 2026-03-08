import pytest
import torch
from search.search_py.nodes import ChanceNode, DecisionNode

# MANDATORY: Module-level pytest marker
pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "bootstrap_method, expected_value",
    [("parent_value", 5.0), ("network_value", 7.0), ("invalid_method", 0.0)],
)
def test_chance_node_bootstrap_methods(bootstrap_method, expected_value):
    # Enforce strict determinism
    torch.manual_seed(42)

    # 1. Setup a Mock Parent (DecisionNode)
    parent = DecisionNode(prior=1.0, parent=None)
    parent.visits = 10
    parent.value_sum = 50.0  # -> parent.value() == 5.0
    parent.network_value = 7.0

    # 2. Configure the ChanceNode class statically (as done in ModularSearch)
    ChanceNode.bootstrap_method = bootstrap_method

    # 3. Initialize the ChanceNode
    chance_node = ChanceNode(prior=1.0, parent=parent)

    # 4. Assert the value method routes to the correct bootstrap calculation
    # Since visits == 0, it must use the configured bootstrap method
    assert chance_node.visits == 0
    assert chance_node.value() == expected_value


def test_chance_node_mu_fpu_bootstrap():
    # Special case for mu_fpu which requires child vector stats
    parent = DecisionNode(prior=1.0, parent=None)
    parent.child_visits = torch.tensor([5.0, 5.0])
    parent.child_values = torch.tensor([10.0, 20.0])  # Weighted avg should be 15.0

    ChanceNode.bootstrap_method = "mu_fpu"
    chance_node = ChanceNode(prior=1.0, parent=parent)

    assert chance_node.value() == 15.0
