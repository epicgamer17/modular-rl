import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional
from ..utils import add_dirichlet_noise


# TODO: make this work with alternating and single player games. also make work for catan (inconsistent turn ordering, ie p1 twice then p2 3 times, then p3 once)
# TODO: make it work with more than 2 players
def backpropagate_(
    tree: TensorDict,
    trajectory: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    leaf_value: torch.Tensor,
):
    """
    Backpropagates the leaf value up the search trajectory.

    Args:
        tree: The MCTS tree TensorDict.
        trajectory: List of (node_idx, action_idx, mask) tuples.
        leaf_value: The predicted value of the leaf node [B].
    """
    batch_size = tree.batch_size[0]
    device = tree.device
    batch_range = torch.arange(batch_size, device=device)

    # Running value starts at the evaluated leaf node value
    running_value = leaf_value.clone()

    # Iterate backwards through the trajectory
    for node_idx, action_idx, mask in reversed(trajectory):
        # 1. Select only active elements for this step
        # This prevents over-counting visits for elements that hit a leaf early.
        b_idx = batch_range[mask]
        n_idx = node_idx[mask]
        a_idx = action_idx[mask]

        if b_idx.numel() == 0:
            continue

        # 2. Fetch reward and discounts
        rewards = tree["children_rewards"][b_idx, n_idx, a_idx]
        discounts = tree["children_discounts"][b_idx, n_idx, a_idx]

        # 3. Accumulate discounted return: G = R + discounts * V
        # TODO: If turn flips/negamax are handled by recurrent_fn, discount can be negative, for now we handle it here.
        step_return = rewards + discounts * running_value[b_idx]
        running_value[b_idx] = step_return

        # 4. Update child/edge visit counts and Q-values at (parent, action)
        tree["children_visits"][b_idx, n_idx, a_idx] += 1
        n_visits = tree["children_visits"][b_idx, n_idx, a_idx].to(
            tree["children_values"].dtype
        )

        old_q = tree["children_values"][b_idx, n_idx, a_idx]
        new_q = old_q + (step_return - old_q) / n_visits
        tree["children_values"][b_idx, n_idx, a_idx] = new_q

        # 5. Update target child node statistics if the node exists
        child_node_idx = tree["children_index"][b_idx, n_idx, a_idx]
        valid_child_mask = child_node_idx >= 0

        if valid_child_mask.any():
            cb_idx = b_idx[valid_child_mask]
            cn_idx = child_node_idx[valid_child_mask]

            tree["node_visits"][cb_idx, cn_idx] += 1
            c_visits = tree["node_visits"][cb_idx, cn_idx].to(tree["node_values"].dtype)

            old_node_v = tree["node_values"][cb_idx, cn_idx]
            new_node_v = (
                old_node_v + (step_return[valid_child_mask] - old_node_v) / c_visits
            )
            tree["node_values"][cb_idx, cn_idx] = new_node_v

        # 6. Update Min-Max Q-value bounds for qtransforms
        tree["min_q"][b_idx] = torch.minimum(tree["min_q"][b_idx], new_q)
        tree["max_q"][b_idx] = torch.maximum(tree["max_q"][b_idx], new_q)
