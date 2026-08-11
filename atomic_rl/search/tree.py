import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional
from ..utils import add_dirichlet_noise


# TODO: should we do something like our replay buffer. Like an init_mcts_tree and then init_stochastic, init_sampled_muzero, etc etc? Or leave it as is?
def init_mcts_tree(
    root_embeddings: torch.Tensor,
    root_logits: torch.Tensor,
    root_value: torch.Tensor,
    num_simulations: int,
    num_actions: int,
    legal_mask: Optional[torch.Tensor] = None,
) -> TensorDict:
    """Initializes the MCTS tree structure as a TensorDict.

    Args:
        root_embeddings: Initial state representations [B, D]
        root_logits: Initial policy logits at the root [B, num_actions]
        root_value: Initial value estimate at the root [B]
        num_simulations: Number of simulations to perform.
        num_actions: Number of possible actions.
        legal_mask: Boolean mask of illegal actions at root [B,
          num_actions]

    Returns:
        A TensorDict representing the initial tree state.
    """
    batch_size = root_embeddings.shape[0]
    max_nodes = num_simulations + 1  # Root + 1 node per simulation

    # We pre-allocate the tree to avoid dynamic resizing (Torch Compile friendly)
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
            "parents": root_embeddings.new_zeros(
                (batch_size, max_nodes), dtype=torch.long
            ),
            "action_from_parent": root_embeddings.new_zeros(
                (batch_size, max_nodes), dtype=torch.long
            ),
            # Structural edges [B, max_nodes, num_actions]
            "children_index": root_embeddings.new_full(
                (batch_size, max_nodes, num_actions), -1, dtype=torch.long
            ),
            "children_prior_logits": root_embeddings.new_zeros(
                (batch_size, max_nodes, num_actions)
            ),
            "children_visits": root_embeddings.new_zeros(
                (batch_size, max_nodes, num_actions)
            ),
            "children_rewards": root_embeddings.new_zeros(
                (batch_size, max_nodes, num_actions)
            ),
            "children_discounts": root_embeddings.new_ones(
                (batch_size, max_nodes, num_actions)
            ),  # TODO: should discounts handle to play stuff? Does that even work?
            "children_values": root_embeddings.new_zeros(
                (batch_size, max_nodes, num_actions)
            ),
            # Model & Environment States
            "embeddings": root_embeddings.new_zeros(
                (batch_size, max_nodes, *root_embeddings.shape[1:])
            ),
            "node_type": root_embeddings.new_zeros(
                (batch_size, max_nodes), dtype=torch.int8
            ),  # 0: decision, 1: chance
            # TODO: remove to_play and is_terminal. Have the dynamics_fn output the correct value for discount factor isntead. 0.0 for terminal states, * -1 if another player otherwise * 1 for discount.
            "to_play": root_embeddings.new_zeros(
                batch_size, max_nodes, dtype=torch.long
            ),
            "is_terminal": root_embeddings.new_zeros(
                (batch_size, max_nodes), dtype=torch.bool
            ),
            "node_counts": root_embeddings.new_ones(batch_size, dtype=torch.long),
            # NOTE: not in mctx?
            # Search bounds for Q-transforms
            # TODO: why init with float inf and -inf and then just reinit below with root value?
            "min_q": root_embeddings.new_full((batch_size,), float("inf")),
            "max_q": root_embeddings.new_full((batch_size,), float("-inf")),
            # Root-specific masks (for Gumbel MuZero and Sampled MuZero, though might need to be extended tree wide for Sampled MuZero)
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

    # Initialize root (index 0)
    tree["embeddings"][:, 0] = root_embeddings
    tree["raw_values"][:, 0] = root_value
    tree["node_values"][:, 0] = root_value
    tree["children_prior_logits"][:, 0] = root_logits

    # # Initialize min_q and max_q with root values
    # tree["min_q"] = root_value.clone()
    # tree["max_q"] = root_value.clone()
    return tree
