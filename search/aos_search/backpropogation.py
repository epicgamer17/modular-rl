"""Pure, vectorized backpropagation functions for the FlatTree MCTS.

Each function accepts a fixed set of **batched tensors** describing a single
depth-level of the backward pass and applies its update using ``torch.where``
— no Python loops over the batch dimension.

Signature contract
------------------
Every backprop function must accept the following arguments and return
``new_values`` (the discounted return to pass to the shallower level)::

    def my_backprop(
        tree:           FlatTree,
        batch_idx:      torch.Tensor,   # [B]  long
        nodes_at_d:     torch.Tensor,   # [B]  int32 — parent nodes
        actions_at_d:   torch.Tensor,   # [B]  int32 — actions from parents
        current_values: torch.Tensor,   # [B]  float32 — return from below
        discount:       float,
        valid_mask:     torch.Tensor,   # [B]  bool
    ) -> torch.Tensor:                  # [B]  float32 — return to pass up

The caller (``_backpropagate`` in ``batched_mcts.py``) is responsible for
computing ``current_values = r + γ * G_{d+1}`` before passing in.
"""

from __future__ import annotations

from typing import Callable

import torch

from search.aos_search.tree import FlatTree


# ---------------------------------------------------------------------------
# Type alias for backprop functions
# ---------------------------------------------------------------------------

BackpropFn = Callable[
    [
        FlatTree,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
        torch.Tensor,
    ],
    torch.Tensor,
]
"""Callable type for a single-depth backpropagation update."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _update_child_visits_and_values_mean(
    tree: FlatTree,
    batch_idx: torch.Tensor,
    parent_long: torch.Tensor,
    action_long: torch.Tensor,
    target_q: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    """Increment child visit count and update Q via incremental mean.

    Args:
        tree: FlatTree to update in-place.
        batch_idx: ``[B]`` long batch indices.
        parent_long: ``[B]`` long parent node indices.
        action_long: ``[B]`` long action indices.
        target_q: ``[B]`` float32 target Q-value.
        valid_mask: ``[B]`` bool mask — only update where True.
    """
    current_visits = tree.children_visits[batch_idx, parent_long, action_long]
    new_visits = current_visits + valid_mask.to(torch.int32)
    tree.children_visits[batch_idx, parent_long, action_long] = new_visits

    old_q = tree.children_values[batch_idx, parent_long, action_long]
    safe_n = new_visits.float().clamp(min=1.0)
    new_q = old_q + valid_mask.float() * (target_q - old_q) / safe_n
    tree.children_values[batch_idx, parent_long, action_long] = new_q


def _update_node_visits_and_values_mean(
    tree: FlatTree,
    batch_idx: torch.Tensor,
    parent_long: torch.Tensor,
    target_v: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    """Increment node visit count and update node value via incremental mean.

    Args:
        tree: FlatTree to update in-place.
        batch_idx: ``[B]`` long batch indices.
        parent_long: ``[B]`` long parent node indices.
        target_v: ``[B]`` float32 target node value.
        valid_mask: ``[B]`` bool mask — only update where True.
    """
    tree.node_visits[batch_idx, parent_long] += valid_mask.to(torch.int32)
    old_v = tree.node_values[batch_idx, parent_long]
    new_v_count = tree.node_visits[batch_idx, parent_long].float().clamp(min=1.0)
    new_v = old_v + valid_mask.float() * (target_v - old_v) / new_v_count
    tree.node_values[batch_idx, parent_long] = new_v


# ---------------------------------------------------------------------------
# average_discounted_backprop
# ---------------------------------------------------------------------------


def average_discounted_backprop(
    tree: FlatTree,
    batch_idx: torch.Tensor,
    nodes_at_d: torch.Tensor,
    actions_at_d: torch.Tensor,
    current_values: torch.Tensor,
    discount: float,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Standard MuZero average-discounted-return backpropagation.

    Updates Q(s, a) and V(s) for the parent node using an incremental mean
    of discounted Monte-Carlo returns.  Rewards are accumulated by the caller
    via ``current_values = r + γ * G_{d+1}``; here we just average them.

    Args:
        tree: FlatTree mutated in-place.
        batch_idx: ``[B]`` long — arange over the batch.
        nodes_at_d: ``[B]`` int32 — parent node indices at this depth.
        actions_at_d: ``[B]`` int32 — actions taken from the parent.
        current_values: ``[B]`` float32 — discounted return ``G = r + γ·G``.
        discount: MDP discount factor γ (unused here; return is pre-discounted
            by the caller in ``_backpropagate``).
        valid_mask: ``[B]`` bool — only update where True.

    Returns:
        ``current_values`` unchanged (the caller passes it further up as-is).
    """
    parent_long = nodes_at_d.long()
    action_long = actions_at_d.long()

    # Q update (child edge statistics)
    _update_child_visits_and_values_mean(
        tree, batch_idx, parent_long, action_long, current_values, valid_mask
    )

    # V update (node statistics)
    _update_node_visits_and_values_mean(
        tree, batch_idx, parent_long, current_values, valid_mask
    )

    return current_values


# ---------------------------------------------------------------------------
# minimax_backprop
# ---------------------------------------------------------------------------


def minimax_backprop(
    tree: FlatTree,
    batch_idx: torch.Tensor,
    nodes_at_d: torch.Tensor,
    actions_at_d: torch.Tensor,
    current_values: torch.Tensor,
    discount: float,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Minimax backpropagation for two-player zero-sum games.

    Uses the Q-values of **all siblings** (not just the path action) to set
    the parent node's value to the correct minimax aggregate.

    Note: Every player is a maximiser because Q-values are stored relative
    to the acting player (acting player's perspective).

    Steps:
      1. Update ``children_visits`` and ``children_values`` (incremental mean)
         for the edge on the traversed path.
      2. Gather sibling Q-values ``tree.children_values[batch_idx, parent]``.
      3. Build a legality mask from ``tree.children_action_mask``.  Visited
         siblings are considered; unvisited but valid ones are excluded from
         aggregation (they have no reliable Q yet — masked via
         ``children_visits == 0``).
      4. Max over siblings.
      5. Return the best_q to propagate upward.

    Args:
        tree: FlatTree mutated in-place.
        batch_idx: ``[B]`` long.
        nodes_at_d: ``[B]`` int32 — parent node indices.
        actions_at_d: ``[B]`` int32 — actions taken from the parent.
        current_values: ``[B]`` float32 — discounted return from child layer.
        discount: MDP discount factor γ (applied by the caller).
        valid_mask: ``[B]`` bool — only update where True.

    Returns:
        ``best_q`` per batch item — the value to propagate upward.
    """
    parent_long = nodes_at_d.long()
    action_long = actions_at_d.long()

    # Step 1: Update edge Q via incremental mean
    _update_child_visits_and_values_mean(
        tree, batch_idx, parent_long, action_long, current_values, valid_mask
    )

    # Step 2: Gather all sibling Q-values and masks
    # [B, max_edges]
    sibling_q = tree.children_values[batch_idx, parent_long]
    sibling_valid = tree.children_action_mask[batch_idx, parent_long]  # [B, E] bool
    sibling_visited = tree.children_visits[batch_idx, parent_long] > 0  # [B, E] bool

    # Only aggregate over valid AND visited siblings (unvisited have no Q yet)
    consider = sibling_valid & sibling_visited  # [B, E] bool

    # Step 3: Compute the best Q over siblings.
    # Since Q-values are stored relative to the acting player,
    # every player is a maximiser of their own Q.
    INF = float("inf")
    q_for_max = torch.where(consider, sibling_q, torch.full_like(sibling_q, -INF))
    best_q = q_for_max.max(dim=-1).values  # [B]

    # Fall back to current node value when no sibling has been visited yet
    any_considered = consider.any(dim=-1)  # [B] bool
    effective_v = torch.where(
        any_considered, best_q, tree.node_values[batch_idx, parent_long]
    )

    # Step 4: Write node stats
    tree.node_visits[batch_idx, parent_long] += valid_mask.to(torch.int32)
    tree.node_values[batch_idx, parent_long] = torch.where(
        valid_mask, effective_v, tree.node_values[batch_idx, parent_long]
    )

    # Step 5: Propagate best_q upward (sign is already correct —
    # the tree stores Q from the perspective of the node's own player)
    return torch.where(valid_mask, best_q, current_values)
