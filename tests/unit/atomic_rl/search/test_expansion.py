import pytest
import torch

from atomic_rl.search import expand_node_, init_mcts_tree

pytestmark = pytest.mark.unit


def test_expand_node():
    """Verify tree expansion correctly links parents to children and registers state transitions."""
    batch_size = 1
    num_simulations = 2
    num_actions = 2
    root_embeddings = torch.zeros(batch_size, 4)
    root_logits = torch.zeros(batch_size, num_actions)
    root_value = torch.zeros(batch_size)

    tree = init_mcts_tree(root_embeddings, root_logits, root_value, num_simulations, num_actions)

    parent_nodes = torch.tensor([0])
    actions_taken = torch.tensor([1])
    policy_logits = torch.tensor([[0.0, 0.0]])  # Softmax will give [0.5, 0.5]
    values = torch.tensor([0.0])
    rewards = torch.tensor([1.5])
    discounts = torch.tensor([0.0])  # is_terminal = (discounts == 0.0)
    next_embeddings = torch.ones(batch_size, 4)

    # Run expansion
    expand_node_(
        tree,
        parent_nodes,
        actions_taken,
        policy_logits,
        values,
        rewards,
        discounts,
        next_embeddings,
    )

    # Assert structural mutations
    assert tree["children_index"][0, 0, 1].item() == 1  # Parent points to child index 1
    assert tree["node_counts"][0].item() == 2  # Total nodes incremented to 2
    torch.testing.assert_close(tree["embeddings"][0, 1], next_embeddings[0])
    assert tree["children_rewards"][0, 0, 1].item() == 1.5
    assert tree["is_terminal"][0, 1].item() is True
    torch.testing.assert_close(
        torch.softmax(tree["children_prior_logits"][0, 1], dim=-1),
        torch.tensor([0.5, 0.5]),
    )