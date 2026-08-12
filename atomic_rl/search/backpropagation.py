import torch
from tensordict import TensorDict
from typing import List, Tuple


# TODO: make this work with alternating and single player games. also make work for catan (inconsistent turn ordering, ie p1 twice then p2 3 times, then p3 once)
# TODO: make it work with more than 2 players
def backpropagate_(
    tree: TensorDict,
    trajectory: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    leaf_value: torch.Tensor,
):
    """
    Backpropagates the leaf value up the search trajectory in-place.

    Matches mctx's `search.backward`: each edge (s_t, a_t) stores the value of
    its child node, so that Q(s, a) = reward(s, a) + discount(s, a) * V(s') is
    reconstructed by `get_qvalues`. In particular, an alternating zero-sum game
    is encoded by a negative discount on the edge, which flips the perspective
    exactly once (in `get_qvalues`), not twice.

    Args:
        tree: The MCTS tree TensorDict matching mctx structure.
        trajectory: List of (node_idx, action_idx, mask) tuples from root down to leaf.
        leaf_value: The predicted value of the leaf node [B].
    """
    batch_size = tree.batch_size[0]
    device = tree.device
    batch_range = torch.arange(batch_size, device=device)

    # Fail fast on mismatched input shapes before doing any tree work.
    assert tree["node_visits"].ndim == 2, (
        f"tree node buffers must be flat [B, N], got shape "
        f"{tuple(tree['node_visits'].shape)}"
    )
    assert leaf_value.shape == (batch_size,), (
        f"leaf_value shape mismatch: expected [{batch_size}], got {tuple(leaf_value.shape)}"
    )

    # Running return starts at the evaluated leaf node value.
    # The leaf node itself was initialized by expand_node_ (node_visits=1,
    # node_values=raw value), matching mctx's update_tree_node, so backprop does
    # not touch it again.
    running_value = leaf_value.clone()

    # Iterate backwards through trajectory (leaf -> root)
    for node_idx, action_idx, mask in reversed(trajectory):
        b_idx = batch_range[mask]
        n_idx = node_idx[mask]
        a_idx = action_idx[mask]

        if b_idx.numel() == 0:
            continue

        # Fetch child node id for the transition (n_idx, a_idx); -1 if the edge
        # has not been expanded this simulation (e.g. terminal lanes).
        child_idx = tree["children_index"][b_idx, n_idx, a_idx]
        has_child = child_idx >= 0

        if has_child.any():
            cb_idx = b_idx[has_child]
            cn_idx = n_idx[has_child]
            ca_idx = a_idx[has_child]
            cc_idx = child_idx[has_child]

            # Store the child node value on the edge (mctx semantics), so that
            # get_qvalues computes Q(s, a) = reward + discount * V(s').
            tree["children_values"][cb_idx, cn_idx, ca_idx] = tree["node_values"][
                cb_idx, cc_idx
            ]

        # Fetch reward and discount for transition from (n_idx, a_idx)
        rewards = tree["children_rewards"][b_idx, n_idx, a_idx]
        discounts = tree["children_discounts"][b_idx, n_idx, a_idx]

        # Accumulate return for parent node s_t: G_t = R_t + discount_t * G_{t+1}
        step_return = rewards + discounts * running_value[b_idx]
        running_value[b_idx] = step_return

        # Update parent node (s_t) statistics (includes Root node at t=0)
        tree["node_visits"][b_idx, n_idx] += 1
        n_visits = tree["node_visits"][b_idx, n_idx].to(tree["node_values"].dtype)

        old_node_v = tree["node_values"][b_idx, n_idx]
        tree["node_values"][b_idx, n_idx] += (step_return - old_node_v) / n_visits

        # Update child edge (s_t, a_t) visit statistics
        tree["children_visits"][b_idx, n_idx, a_idx] += 1
