import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional
from ..utils import add_dirichlet_noise


# TODO: make this work with alternating and single player games. also make work for catan (inconsistent turn ordering, ie p1 twice then p2 3 times, then p3 once)
# TODO: make it work with more than 2 players
import torch
from tensordict import TensorDict
from typing import List, Tuple


def backpropagate_(
    tree: TensorDict,
    trajectory: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    leaf_value: torch.Tensor,
):
    """
    Backpropagates the leaf value up the search trajectory in-place.

    Args:
        tree: The MCTS tree TensorDict matching mctx structure.
        trajectory: List of (node_idx, action_idx, mask) tuples from root down to leaf.
        leaf_value: The predicted value of the leaf node [B].
    """
    batch_size = tree.batch_size[0]
    device = tree.device
    batch_range = torch.arange(batch_size, device=device)

    # Running return starts at the evaluated leaf node value
    running_value = leaf_value.clone()

    # 1. Update leaf node statistics (s_L) before stepping backward
    if trajectory:
        last_node_idx, last_action_idx, last_mask = trajectory[-1]
        b_idx = batch_range[last_mask]
        ln_idx = last_node_idx[last_mask]
        la_idx = last_action_idx[last_mask]

        leaf_child_idx = tree["children_index"][b_idx, ln_idx, la_idx]
        valid_leaf_mask = leaf_child_idx >= 0

        if valid_leaf_mask.any():
            lb_idx = b_idx[valid_leaf_mask]
            lc_idx = leaf_child_idx[valid_leaf_mask]

            # Increment leaf node visits
            tree["node_visits"][lb_idx, lc_idx] += 1
            l_visits = tree["node_visits"][lb_idx, lc_idx].to(tree["node_values"].dtype)

            # Update leaf node value towards leaf_value
            old_leaf_v = tree["node_values"][lb_idx, lc_idx]
            target_leaf_v = running_value[lb_idx][valid_leaf_mask]
            tree["node_values"][lb_idx, lc_idx] += (
                target_leaf_v - old_leaf_v
            ) / l_visits

    # 2. Iterate backwards through trajectory (leaf -> root)
    for node_idx, action_idx, mask in reversed(trajectory):
        b_idx = batch_range[mask]
        n_idx = node_idx[mask]
        a_idx = action_idx[mask]

        if b_idx.numel() == 0:
            continue

        # Fetch reward and discount for transition from (n_idx, a_idx)
        rewards = tree["children_rewards"][b_idx, n_idx, a_idx]
        discounts = tree["children_discounts"][b_idx, n_idx, a_idx]

        # Accumulate return for parent node s_t: G_t = R_t + discount_t * G_{t+1}
        step_return = rewards + discounts * running_value[b_idx]
        running_value[b_idx] = step_return

        # 3. Update parent node (s_t) statistics (includes Root node at t=0)
        tree["node_visits"][b_idx, n_idx] += 1
        n_visits = tree["node_visits"][b_idx, n_idx].to(tree["node_values"].dtype)

        old_node_v = tree["node_values"][b_idx, n_idx]
        tree["node_values"][b_idx, n_idx] += (step_return - old_node_v) / n_visits

        # 4. Update child edge (s_t, a_t) Q-value and visit statistics
        tree["children_visits"][b_idx, n_idx, a_idx] += 1
        c_visits = tree["children_visits"][b_idx, n_idx, a_idx].to(
            tree["children_values"].dtype
        )

        old_q = tree["children_values"][b_idx, n_idx, a_idx]
        new_q = old_q + (step_return - old_q) / c_visits
        tree["children_values"][b_idx, n_idx, a_idx] = new_q

        # 5. Update Min-Max Q-value bounds
        tree["min_q"][b_idx] = torch.minimum(tree["min_q"][b_idx], new_q)
        tree["max_q"][b_idx] = torch.maximum(tree["max_q"][b_idx], new_q)
