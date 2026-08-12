import torch
import torch.nn.functional as F
from tensordict import TensorDict
from typing import Tuple, Callable, List
from .qtransforms import qtransform_by_parent_and_siblings


# TODO: should we merge this with select_leaf?
def puct_score(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
    depth: int,
    *,
    pb_c_init: float = 1.25,
    pb_c_base: float = 19652.0,
    qtransform: Callable = qtransform_by_parent_and_siblings,
    **kwargs,
) -> torch.Tensor:
    """
    Computes PUCT scores matching mctx.muzero_action_selection.

    Formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    """
    batch_size = tree.batch_size[0]
    batch_range = torch.arange(batch_size, device=tree.device)

    assert tree["node_visits"].ndim == 2, (
        f"tree node buffers must be flat [B, N], got shape "
        f"{tuple(tree['node_visits'].shape)}"
    )
    assert parent_nodes.shape == (batch_size,), (
        f"parent_nodes shape mismatch: expected [{batch_size}], got {tuple(parent_nodes.shape)}"
    )
    assert isinstance(depth, int) and depth >= 0, (
        f"depth must be a non-negative int, got {depth!r}"
    )
    assert pb_c_init >= 0, f"pb_c_init must be >= 0, got {pb_c_init}"
    assert pb_c_base >= 0, f"pb_c_base must be >= 0, got {pb_c_base}"

    # 1. Fetch node and child statistics
    visit_counts = tree["children_visits"][
        batch_range, parent_nodes
    ]  # [B, num_actions]
    node_visits = tree["node_visits"][batch_range, parent_nodes].unsqueeze(-1)  # [B, 1]
    prior_logits = tree["children_prior_logits"][
        batch_range, parent_nodes
    ]  # [B, num_actions]
    prior_probs = F.softmax(prior_logits, dim=-1)

    # 2. Compute PUCT exploration penalty
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
    scores = value_score + policy_score + noise

    # 5. Mask root invalid actions at depth 0
    if depth == 0 and "root_legal_mask" in tree.keys():
        legal_mask = tree["root_legal_mask"]  # [B, num_actions]
        min_val = torch.finfo(scores.dtype).min
        scores = torch.where(legal_mask, scores, min_val)

    return scores


# TODO: work with Batched MCTS, batch_mcts.pdf
# TODO: work with Vectorized MCTS
# TODO: work iwth Batched + Vectorized MCTS
# TODO: Stochastic MuZero
def select_leaf(
    tree: TensorDict,
    pb_c_base: float = 19652.0,
    pb_c_init: float = 1.25,
    qtransform: Callable = qtransform_by_parent_and_siblings,
    scoring_fn: Callable = puct_score,
    max_depth: int = 512,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    """Selects leaf nodes to expand by descending the search tree.

    Returns:
        leaf_parents: Parent node indices [B] where expansion occurs.
        leaf_actions: Actions taken [B] from leaf_parents.
        expansion_mask: Mask [B] indicating which batch items require expansion (True)
          vs items that hit terminal nodes or max depth (False).
        trajectory: List of (node_idx, action_idx, active_mask) tuples.
    """
    batch_size = tree.batch_size[0]
    device = tree.device
    batch_range = torch.arange(batch_size, device=device)

    assert tree["node_visits"].ndim == 2, (
        f"tree node buffers must be flat [B, N], got shape "
        f"{tuple(tree['node_visits'].shape)}"
    )
    assert isinstance(max_depth, int) and max_depth >= 1, (
        f"max_depth must be an int >= 1, got {max_depth!r}"
    )
    assert pb_c_init >= 0, f"pb_c_init must be >= 0, got {pb_c_init}"
    assert pb_c_base >= 0, f"pb_c_base must be >= 0, got {pb_c_base}"

    current_node = torch.zeros(batch_size, dtype=torch.long, device=device)
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    leaf_parents = torch.zeros(batch_size, dtype=torch.long, device=device)
    leaf_actions = torch.zeros(batch_size, dtype=torch.long, device=device)
    expansion_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    trajectory = []

    for depth in range(max_depth):
        # 1. Stop descent if node is terminal
        is_term = tree["is_terminal"][batch_range, current_node]
        active_mask = active_mask & (~is_term)

        if not active_mask.any():
            break

        # 2. Score actions for active parents
        scores = scoring_fn(
            tree,
            current_node,
            depth=depth,
            pb_c_base=pb_c_base,
            pb_c_init=pb_c_init,
            qtransform=qtransform,
        )  # [B, A]

        action = scores.argmax(dim=-1)  # [B]

        # 3. Record trajectory step for active items
        trajectory.append((current_node.clone(), action.clone(), active_mask.clone()))

        # 4. Check if child is an unvisited leaf (-1)
        next_node = tree["children_index"][batch_range, current_node, action]
        is_leaf = next_node == -1

        # 5. Capture exact expansion targets for items hitting an unvisited leaf
        newly_found_leaf = active_mask & is_leaf
        if newly_found_leaf.any():
            leaf_parents = torch.where(newly_found_leaf, current_node, leaf_parents)
            leaf_actions = torch.where(newly_found_leaf, action, leaf_actions)
            expansion_mask = expansion_mask | newly_found_leaf

        # 6. Deactivate items reaching leaf
        active_mask = active_mask & (~is_leaf)

        # 7. Advance current_node for items still actively descending
        current_node = torch.where(active_mask, next_node, current_node)

    return leaf_parents, leaf_actions, expansion_mask, trajectory
