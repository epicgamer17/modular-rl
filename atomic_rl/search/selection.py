import torch
import torch.nn.functional as F
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional
from ..utils import add_dirichlet_noise
from .qtransforms import (
    qtransform_completed_by_mix_value,
    qtransform_by_parent_and_siblings,
    qtransform_by_min_max,
)


# TODO: should we merge this with select_leaf?
def puct_score(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
    depth: int,
    *,
    pb_c_init: float = 1.25,
    pb_c_base: float = 19652.0,
    qtransform: Callable = qtransform_by_parent_and_siblings,
) -> torch.Tensor:
    """
    The PUCT score.

    The formula is: Q(s,a) + U(s,a)
    where U(s,a) = C * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

    Args:
        q_values: Q-values of the actions.
        policy_prior: Prior probabilities of the actions. Not logits!
        visit_counts: Visit counts for each action.
        total_visit_counts: Total visit count for the parent state.
        pb_c_base: Base constant for PUCT.
        pb_c_init: Additive constant for PUCT (used for virtual exploration).
    """
    # Fail Fast: Ensure shape contracts match expected [B, num_actions] and [B, 1] dimensions
    # TODO: Add shape asserts

    batch_size = tree.batch_size[0]
    batch_range = torch.arange(batch_size, device=tree.device)

    # 1. Fetch node and child statistics
    visit_counts = tree["children_visits"][
        batch_range, parent_nodes
    ]  # [B, num_actions]
    node_visits = tree["node_visits"][batch_range, parent_nodes].unsqueeze(-1)  # [B, 1]
    prior_logits = tree["children_prior_logits"][
        batch_range, parent_nodes
    ]  # [B, num_actions]
    prior_probs = F.softmax(prior_logits, dim=-1)

    # 2. Compute PUCT exploration term: c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    pb_c = pb_c_init + torch.log((node_visits + pb_c_base + 1.0) / pb_c_base)
    policy_score = (
        torch.sqrt(node_visits.to(prior_probs.dtype))
        * pb_c
        * prior_probs
        / (visit_counts + 1.0)
    )

    # 3. Compute transformed Q-value score
    value_score = qtransform(tree, parent_nodes)  # [B, num_actions]

    # 4. Add tiny uniform noise for tie breaking (matches mctx)
    noise = 1e-7 * torch.rand_like(value_score)

    puct_score = value_score + policy_score + noise

    # 5. Mask root invalid actions if at depth 0
    if depth == 0 and "root_legal_mask" in tree.keys():
        legal_mask = tree["root_legal_mask"]  # [B, num_actions]
        # Mask out where legal_mask is False
        puct_score = torch.where(legal_mask, puct_score, -float("inf"))

    return puct_score


def gumbel_interior_action_score(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
    depth: int = 0,
    *,
    qtransform: Callable = qtransform_completed_by_mix_value,
) -> torch.Tensor:
    """Computes deterministic argmax input for non-root nodes in Gumbel MuZero.

    Matches `mctx.gumbel_muzero_interior_action_selection`.
    """
    pass


# TODO: work with Batched MCTS, batch_mcts.pdf
# TODO: work with Vectorized MCTS
# TODO: work iwth Batched + Vectorized MCTS
# TODO: can we reuse our action_selection.py methods/functions?
# TODO: Stochastic MuZero
def select_leaf(
    tree: TensorDict,
    pb_c_base: float = 19652.0,
    pb_c_init: float = 1.25,
    qtransform: Callable = qtransform_by_parent_and_siblings,
    scoring_fn: Callable = puct_score,
    max_depth: int = 512,
) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """
    Selects leaf nodes to expand by descending the search tree using scoring_fn.

    Args:
        tree: The MCTS tree TensorDict.
        pb_c_base: PUCT base constant.
        pb_c_init: PUCT init constant.
        qtransform: Q-value normalization function.
        scoring_fn: Function (tree, parent_nodes, depth, ...) -> scores [B, A].
        max_depth: Maximum tree descent depth.

    Returns:
        A tuple containing:
            - leaf_indices: The selected leaf node indices [B].
            - trajectory: List of (node_idx, action_idx, active_mask) tuples.
    """
    batch_size = tree.batch_size[0]
    device = tree.device
    batch_range = torch.arange(batch_size, device=device)

    current_node = tree["min_q"].new_zeros(batch_size, dtype=torch.long)
    trajectory = []  # List of (node_idx, action_idx, mask)

    # Active batch mask (True = still descending, False = hit leaf or terminal)
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    # The search depth is naturally bounded by the number of nodes or a safety limit
    for depth in range(max_depth):
        # 1. Compute action scores using decoupled scoring function
        scores = scoring_fn(
            tree,
            current_node,
            depth=depth,
            pb_c_base=pb_c_base,
            pb_c_init=pb_c_init,
            qtransform=qtransform,
        )  # [B, num_actions]

        # 2. Select best action
        # TODO: should we use action_selection.py here? Does it make sense to use it?
        action = scores.argmax(dim=-1)  # [B]

        # 3. Check for leaf (child index is -1) or terminal node
        next_node = tree["children_index"][batch_range, current_node, action]
        is_leaf = next_node == -1
        is_term = tree["is_terminal"][batch_range, current_node]

        # 5. Record to trajectory (only for elements that were active at the START of this step)
        trajectory.append((current_node.clone(), action.clone(), active_mask.clone()))

        # 6. Update active mask: those who hit a leaf or terminal node are no longer active for the NEXT step
        active_mask = active_mask & (~is_leaf) & (~is_term)

        if not active_mask.any():
            break

        # Update current_node for elements that haven't hit a leaf/terminal
        current_node = torch.where(active_mask, next_node, current_node)

    return current_node, trajectory
