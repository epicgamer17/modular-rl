import torch
from tensordict import TensorDict
from typing import Optional


# TODO: should we do something like our replay buffer. Like an init_mcts_tree and then init_stochastic, init_sampled_muzero, etc etc? Or leave it as is?
def init_mcts_tree(
    root_embeddings: torch.Tensor,
    root_logits: torch.Tensor,
    root_value: torch.Tensor,
    num_simulations: int,
    num_actions: int,
    legal_mask: Optional[torch.Tensor] = None,  # [B, num_actions] True for legal
) -> TensorDict:
    """Initializes the MCTS tree structure as a TensorDict matching mctx semantics.

    Args:
        root_embeddings: Initial state representations [B, D]
        root_logits: Initial policy logits at the root [B, num_actions]
        root_value: Initial value estimate at the root [B]
        num_simulations: Number of simulations to perform.
        num_actions: Number of possible actions.
        legal_mask: Boolean mask of legal actions at root [B, num_actions]

    Returns:
        A TensorDict representing the initial tree state.
    """
    assert isinstance(num_simulations, int) and not isinstance(
        num_simulations, bool
    ), f"num_simulations must be an int, got {num_simulations!r}"
    assert num_simulations >= 1, f"num_simulations must be >= 1, got {num_simulations}"
    assert isinstance(num_actions, int) and not isinstance(num_actions, bool), (
        f"num_actions must be an int, got {num_actions!r}"
    )
    assert num_actions >= 1, f"num_actions must be >= 1, got {num_actions}"
    assert root_embeddings.ndim >= 2, (
        f"root_embeddings must be at least 2D [B, D...], got shape "
        f"{tuple(root_embeddings.shape)}"
    )
    batch_size = root_embeddings.shape[0]
    assert root_logits.shape == (batch_size, num_actions), (
        f"root_logits shape mismatch: expected [{batch_size}, {num_actions}], "
        f"got {tuple(root_logits.shape)}"
    )
    assert root_value.shape == (batch_size,), (
        f"root_value shape mismatch: expected [{batch_size}], got {tuple(root_value.shape)}"
    )
    if legal_mask is not None:
        assert legal_mask.shape == (batch_size, num_actions), (
            f"legal_mask shape mismatch: expected [{batch_size}, {num_actions}], "
            f"got {tuple(legal_mask.shape)}"
        )
    max_nodes = num_simulations + 1  # Root + 1 node per simulation

    # Pre-allocate the tree tensors (Torch Compile friendly)
    tree = TensorDict(
        {
            # Node-level statistics [B, max_nodes]
            "node_visits": root_embeddings.new_zeros(
                (batch_size, max_nodes), dtype=torch.long
            ),
            "raw_values": root_embeddings.new_zeros(
                (batch_size, max_nodes), dtype=torch.float32
            ),
            "node_values": root_embeddings.new_zeros(
                (batch_size, max_nodes), dtype=torch.float32
            ),
            # NO_PARENT initialized to -1
            "parents": root_embeddings.new_full(
                (batch_size, max_nodes), -1, dtype=torch.long
            ),
            "action_from_parent": root_embeddings.new_full(
                (batch_size, max_nodes), -1, dtype=torch.long
            ),
            # Structural edges [B, max_nodes, num_actions]
            "children_index": root_embeddings.new_full(
                (batch_size, max_nodes, num_actions), -1, dtype=torch.long
            ),
            "children_prior_logits": root_embeddings.new_zeros(
                (batch_size, max_nodes, num_actions)
            ),
            "children_visits": root_embeddings.new_zeros(
                (batch_size, max_nodes, num_actions), dtype=torch.long
            ),
            "children_rewards": root_embeddings.new_zeros(
                (batch_size, max_nodes, num_actions)
            ),
            "children_discounts": root_embeddings.new_ones(
                (batch_size, max_nodes, num_actions)
            ),
            "children_values": root_embeddings.new_zeros(
                (batch_size, max_nodes, num_actions)
            ),
            # Model & Environment States
            "embeddings": root_embeddings.new_zeros(
                (batch_size, max_nodes, *root_embeddings.shape[1:])
            ),
            "is_terminal": root_embeddings.new_zeros(
                (batch_size, max_nodes), dtype=torch.bool
            ),
            # Node counter starts at 1 (index 0 is root)
            "node_counts": root_embeddings.new_ones(batch_size, dtype=torch.long),
            # Root action mask (mctx uses root_invalid_actions = ~legal_mask)
            "root_legal_mask": (
                legal_mask
                if legal_mask is not None
                else root_embeddings.new_ones(
                    (batch_size, num_actions), dtype=torch.bool
                )
            ),
        },
        batch_size=[batch_size],
    )

    # Mask illegal actions in root prior logits
    masked_root_logits = root_logits.clone()
    if legal_mask is not None:
        min_val = torch.finfo(masked_root_logits.dtype).min
        masked_root_logits = masked_root_logits.masked_fill(~legal_mask, min_val)

    # Populate root node (index 0)
    tree["embeddings"][:, 0] = root_embeddings
    tree["raw_values"][:, 0] = root_value
    tree["node_values"][:, 0] = root_value
    tree["children_prior_logits"][:, 0] = masked_root_logits

    return tree
