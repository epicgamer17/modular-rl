import torch
from tensordict import TensorDict
from typing import Optional


# TODO: legal move masking for AlphaZero and terminal nodes for AlphaZero
# TODO: Sampled MuZero
# TODO: initial value for unvisited nodes, allow options, AlphaZero and MuZero: 0, Gumbel Muzero, v_mix, EfficientZero and Batch MCTS mean score
def expand_node_(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
    actions_taken: torch.Tensor,  # [B]
    policy_logits: torch.Tensor,  # [B, A]
    value: torch.Tensor,  # [B]
    rewards: torch.Tensor,  # [B]
    discounts: torch.Tensor,  # [B]
    next_embeddings: torch.Tensor,  # [B, D...]
    legal_mask: Optional[torch.Tensor] = None,  # [B, A]
    masks: Optional[torch.Tensor] = None,  # [B] boolean mask
):
    """Adds newly evaluated nodes to the MCTS tree in-place."""
    batch_size = tree.batch_size[0]
    device = tree.device
    batch_range = torch.arange(batch_size, device=device)

    # Fail fast on mismatched input shapes before doing any tree work.
    assert tree["node_visits"].ndim == 2, (
        f"tree node buffers must be flat [B, N], got shape "
        f"{tuple(tree['node_visits'].shape)}"
    )
    num_actions = tree["children_index"].shape[-1]

    assert parent_nodes.shape == (batch_size,), (
        f"parent_nodes shape mismatch: expected [{batch_size}], got {tuple(parent_nodes.shape)}"
    )
    assert actions_taken.shape == (batch_size,), (
        f"actions_taken shape mismatch: expected [{batch_size}], got {tuple(actions_taken.shape)}"
    )
    assert policy_logits.shape == (batch_size, num_actions), (
        f"policy_logits shape mismatch: expected [{batch_size}, {num_actions}], "
        f"got {tuple(policy_logits.shape)}"
    )
    assert value.shape == rewards.shape == discounts.shape == (batch_size,), (
        f"value/rewards/discounts must all be [{batch_size}], got "
        f"{tuple(value.shape)}, {tuple(rewards.shape)}, {tuple(discounts.shape)}"
    )

    expected_embed = (batch_size, *tree["embeddings"].shape[2:])
    assert next_embeddings.shape == expected_embed, (
        f"next_embeddings shape {tuple(next_embeddings.shape)} does not match "
        f"expected {expected_embed}"
    )
    if legal_mask is not None:
        assert legal_mask.shape == (batch_size, num_actions), (
            f"legal_mask shape mismatch: expected [{batch_size}, {num_actions}], "
            f"got {tuple(legal_mask.shape)}"
        )
    if masks is not None:
        assert masks.shape == (batch_size,), (
            f"masks shape mismatch: expected [{batch_size}], got {tuple(masks.shape)}"
        )

    # 1. Filter active batch elements if a mask is provided
    if masks is not None:
        b_idx = batch_range[masks]
        p_idx = parent_nodes[masks]
        a_idx = actions_taken[masks]

        # Early return if no batch elements are active
        if b_idx.numel() == 0:
            return

        policy_logits = policy_logits[masks]
        value = value[masks]
        rewards = rewards[masks]
        discounts = discounts[masks]
        next_embeddings = next_embeddings[masks]

        if legal_mask is not None:
            legal_mask = legal_mask[masks]
    else:
        b_idx = batch_range
        p_idx = parent_nodes
        a_idx = actions_taken

    # 2. Allocate new node indices per batch item
    new_node_indices = tree["node_counts"][b_idx]

    # 3. Update structural forward edge: parent -> child
    tree["children_index"][b_idx, p_idx, a_idx] = new_node_indices
    tree["children_rewards"][b_idx, p_idx, a_idx] = rewards
    tree["children_discounts"][b_idx, p_idx, a_idx] = discounts

    # 4. Update structural reverse edge: child -> parent
    tree["parents"][b_idx, new_node_indices] = p_idx
    tree["action_from_parent"][b_idx, new_node_indices] = a_idx

    # 5. Initialize children of the NEW node to UNVISITED (-1)
    tree["children_index"][b_idx, new_node_indices] = -1

    # 6. Reset/initialize search statistics for the new node.
    # The node starts with 1 visit so that the PUCT exploration term
    # sqrt(N(s)) is non-zero when this node is selected in a later
    # simulation (matches mctx.update_tree_node, which sets new_visit=1).
    tree["node_visits"][b_idx, new_node_indices] = 1
    tree["children_visits"][b_idx, new_node_indices] = 0
    tree["children_values"][b_idx, new_node_indices] = 0.0

    # 7. Mask illegal actions safely across float dtypes
    curr_logits = policy_logits.clone()
    if legal_mask is not None:
        min_val = torch.finfo(curr_logits.dtype).min
        curr_logits = curr_logits.masked_fill(~legal_mask, min_val)

    # 8. Populate child node values and embeddings
    tree["embeddings"][b_idx, new_node_indices] = next_embeddings
    tree["children_prior_logits"][b_idx, new_node_indices] = curr_logits
    tree["raw_values"][b_idx, new_node_indices] = value
    tree["node_values"][b_idx, new_node_indices] = value

    tree["is_terminal"][b_idx, new_node_indices] = discounts == 0.0

    # 10. Increment allocation count for active elements
    tree["node_counts"][b_idx] += 1
