import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional
from ..utils import add_dirichlet_noise


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
    next_to_play: Optional[torch.Tensor] = None,  # [B]
    masks: Optional[torch.Tensor] = None,  # [B] boolean mask
):
    """Adds newly evaluated nodes to the MCTS tree in-place."""
    batch_size = tree.batch_size[0]
    device = tree.device
    batch_range = torch.arange(batch_size, device=device)

    # 1. Filter indices up front if a mask is provided
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
        if next_to_play is not None:
            next_to_play = next_to_play[masks]
    else:
        b_idx = batch_range
        p_idx = parent_nodes
        a_idx = actions_taken

    # 2. Get new_node_indices FOR THE MASKED BATCH ELEMENTS (length = len(b_idx))
    new_node_indices = tree["node_counts"][b_idx]

    # 3. Update structural edge from parent -> child
    tree["children_index"][b_idx, p_idx, a_idx] = new_node_indices
    tree["children_rewards"][b_idx, p_idx, a_idx] = rewards
    tree["children_discounts"][b_idx, p_idx, a_idx] = discounts

    # 4. Mask illegal actions in policy logits
    curr_logits = policy_logits.clone()
    if legal_mask is not None:
        curr_logits = curr_logits.masked_fill(~legal_mask, -1e9)

    # 5. Populate child node data (Shapes now match [len(b_idx), 3, 3, 2] == [2, 3, 3, 2])
    tree["embeddings"][b_idx, new_node_indices] = next_embeddings
    tree["children_prior_logits"][b_idx, new_node_indices] = curr_logits
    tree["raw_values"][b_idx, new_node_indices] = value
    tree["node_values"][b_idx, new_node_indices] = value

    # 6. Populate optional attributes
    if next_to_play is not None:
        tree["to_play"][b_idx, new_node_indices] = next_to_play

    tree["is_terminal"][b_idx, new_node_indices] = discounts == 0.0

    # 7. Increment node_counts ONLY for the active batch elements
    tree["node_counts"][b_idx] += 1
