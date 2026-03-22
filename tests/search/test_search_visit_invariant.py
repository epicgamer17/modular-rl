import pytest

pytestmark = pytest.mark.unit

import torch
from search.aos_search.tree import FlatTree
from search.aos_search.batched_mcts import batched_mcts_step
from search.aos_search.backpropogation import average_discounted_backprop
from tests.search.conftest import MockAOSNetwork


def test_node_visit_invariant_collision():
    torch.manual_seed(42)
    """Test that node_visits correctly reflects edge visits even with collisions."""
    B = 1
    num_actions = 4
    device = torch.device("cpu")

    # 1. Setup tree
    tree = FlatTree.allocate(B, 100, num_actions, 1, device)
    tree.node_visits[0, 0] = 1
    tree.node_values[0, 0] = 0.0
    tree.to_play[0, 0] = 0
    tree.children_prior_logits[0, 0, :] = 0.0
    tree.children_action_mask[0, 0, :] = True

    # 2. Run batched_mcts_step with search_batch_size=2
    # We want both simulations to pick the same edge.
    # Since priors and values are uniform, they likely will (especially with the noise we added).
    # To be SURE, we can temporarily monkeypatch scoring to return a constant best action.

    net = MockAOSNetwork()

    # We use num_simulations=2 and search_batch_size=2
    # This will run 1 call to batched_mcts_step with B_search=2

    batched_mcts_step(
        tree,
        net,
        max_depth=1,
        pb_c_init=1.25,
        pb_c_base=19652,
        discount=1.0,
        search_batch_size=2,
        virtual_loss_visits=1.0,
        virtual_loss_value=-1.0,
        backprop_fn=average_discounted_backprop,
    )

    # After 1 step with batch_size 2:
    # Root should have 3 visits (init 1 + 2 sims)
    assert tree.node_visits[0, 0] == 3

    # Find the action taken (should be the same for both if collision occurred)
    root_visits = tree.children_visits[0, 0]
    action = root_visits.argmax().item()
    action_visits = root_visits[action].item()

    # If collision occurred, action_visits should be 2
    # If no collision, they'd be 1 each for two different actions.
    # Given MockNetwork and uniform priors, collision is highly probable.

    child_idx = tree.children_index[0, 0, action].item()
    assert child_idx != -1

    child_node_visits = tree.node_visits[0, child_idx].item()

    print(
        f"Action: {action}, Edge Visits: {action_visits}, Child Node Visits: {child_node_visits}"
    )

    # The invariant: node_visits must match the visits to the edge that points to it
    assert (
        action_visits == child_node_visits
    ), f"Invariant broken: edge visits {action_visits} != node visits {child_node_visits}"


if __name__ == "__main__":
    test_node_visit_invariant_collision()
