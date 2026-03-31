"""Vectorized MCTS scoring operating directly on the FlatTree SoA.

Replaces the OOP class hierarchies in ``search_py/scoring_methods.py`` and
``search_py/search_selectors.py`` with pure-tensor scoring functions that
run across the entire batch in a single call — no Python loops.

Two scoring functions are provided:
  * :func:`ucb_score_fn`    — standard MuZero PUCT UCB.
  * :func:`gumbel_score_fn` — Gumbel MuZero improved-policy scoring.

Both conform to the :data:`ScoringFn` callable protocol and can be swapped
freely inside ``batched_mcts_step``.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from old_muzero.search.aos_search.tree import FlatTree
from old_muzero.search.aos_search.min_max_stats import VectorizedMinMaxStats


# ---------------------------------------------------------------------------
# Callable protocol — any scoring function must match this signature
# ---------------------------------------------------------------------------

ScoringFn = Callable[
    [FlatTree, torch.Tensor, "..."],
    torch.Tensor,
]
"""Scoring-function type: ``(tree, node_indices, ...) -> [B, max_edges]``."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def compute_v_mix(
    tree: FlatTree,
    node_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute the v_mix bootstrap value for a set of nodes.

    Replicates the math from the old ``DecisionNode.get_v_mix``::

        sum_N   = Σ_a N(s, a)
        E_vis   = Σ_{a: visited} π(a) * Q(s, a) / Σ_{a: visited} π(a)
        v_mix   = (V(s) + sum_N * E_vis) / (1 + sum_N)

    When no children have been visited, ``v_mix ≡ V(s)``.

    Args:
        tree: Populated :class:`FlatTree`.
        node_indices: ``[B]`` int32 — node indices to compute v_mix for.

    Returns:
        ``[B]`` float32 — the v_mix values.
    """
    B = node_indices.shape[0]
    batch_idx = torch.arange(B, device=node_indices.device)

    # [B] — use the pristine network estimate v̂_π (Eq. 33).
    # node_values is contaminated by backprop; raw_network_values is immutable.
    parent_values = tree.raw_network_values[batch_idx, node_indices].float()

    # [B, max_edges]
    child_visits = tree.children_visits[batch_idx, node_indices].float()
    child_values = tree.children_values[batch_idx, node_indices].float()
    child_logits = tree.children_prior_logits[batch_idx, node_indices]

    # Prior probabilities (softmax is safe against -inf → 0)
    # [B, max_edges]
    real_edge_mask = tree.children_action_mask[batch_idx, node_indices]
    masked_logits = torch.where(
        real_edge_mask, child_logits, torch.full_like(child_logits, -float("inf"))
    )
    child_priors = torch.softmax(masked_logits, dim=-1)

    # Visited mask
    # [B, max_edges]
    expanded_mask = tree.children_index[batch_idx, node_indices] != -1
    visited = expanded_mask

    # sum_N = total visit count across children  [B]
    sum_N = child_visits.sum(dim=-1)

    # --- E_vis: expected Q over visited actions, weighted by priors ---
    # Mask out unvisited priors / Q-values to avoid polluting the sum
    priors_vis = torch.where(visited, child_priors, torch.zeros_like(child_priors))
    q_vis = torch.where(visited, child_values, torch.zeros_like(child_values))

    # [B] — sum of priors for visited actions
    sum_priors_vis = priors_vis.sum(dim=-1).clamp(min=1e-8)

    # [B] — expected Q = Σ(π * Q) / Σπ
    expected_q_vis = (priors_vis * q_vis).sum(dim=-1) / sum_priors_vis

    # --- v_mix = (V + sum_N * E_vis) / (1 + sum_N) ---
    # When sum_N == 0 (no visited children): v_mix = V / 1 = V
    # [B]
    v_mix = (parent_values + sum_N * expected_q_vis) / (1.0 + sum_N)

    return v_mix


# ---------------------------------------------------------------------------
# ucb_score_fn  (standard MuZero PUCT)
# ---------------------------------------------------------------------------


def ucb_score_fn(
    tree: FlatTree,
    node_indices: torch.Tensor,
    pb_c_init: float,
    pb_c_base: float,
    min_max_stats: Optional[VectorizedMinMaxStats] = None,
    bootstrap_method: str = "parent_value",
    **kwargs,
) -> torch.Tensor:
    """Compute MuZero-style UCB scores for every edge of the selected nodes.

    The standard formula is::

        prior_score  = pb_c * prior_prob
        value_score  = normalized Q(s, a)
        UCB          = prior_score + value_score

    where::

        pb_c = [ log((1 + N(s) + pb_c_base) / pb_c_base) + pb_c_init ]
               * sqrt(N(s)) / (1 + N(s, a))

    For **unvisited** edges (``N(s,a) == 0``) the Q-value is bootstrapped
    from the parent node's value ``V(s)``; that bootstrap is also normalized
    through ``min_max_stats`` if provided.

    Args:
        tree: A fully-allocated :class:`FlatTree` instance.
        node_indices: 1-D ``int32`` tensor of shape ``[B]``.
        pb_c_init: PUCT exploration constant (additive).
        pb_c_base: PUCT exploration constant (base for the log term).
        min_max_stats: Optional :class:`VectorizedMinMaxStats` for global
            Q normalisation.

    Returns:
        UCB scores of shape ``[B, max_edges]``.
    """
    B = node_indices.shape[0]
    batch_idx = torch.arange(B, device=node_indices.device)

    # --- gather per-node stats ------------------------------------------------
    parent_visits = tree.node_visits[batch_idx, node_indices].float()  # [B]
    parent_values = tree.node_values[batch_idx, node_indices]  # [B]

    child_visits = tree.children_visits[batch_idx, node_indices].float()  # [B, E]
    child_values = tree.children_values[batch_idx, node_indices]  # [B, E]
    child_logits = tree.children_prior_logits[batch_idx, node_indices]  # [B, E]
    real_edge_mask = tree.children_action_mask[batch_idx, node_indices]  # [B, E]
    masked_logits = torch.where(
        real_edge_mask, child_logits, torch.full_like(child_logits, -float("inf"))
    )
    child_priors = torch.softmax(masked_logits, dim=-1)  # [B, E]

    # --- PUCT exploration bonus -----------------------------------------------
    pv = parent_visits.unsqueeze(-1)  # [B, 1]
    pb_c = torch.log((1.0 + pv + pb_c_base) / pb_c_base) + pb_c_init  # [B, 1]
    pb_c = pb_c * torch.sqrt(pv) / (1.0 + child_visits)  # [B, E]
    prior_score = pb_c * child_priors  # [B, E]

    # --- Q-value (bootstrap unvisited from parent V) --------------------------
    expanded_mask = tree.children_index[batch_idx, node_indices] != -1
    visited_mask = expanded_mask

    if bootstrap_method == "parent_value":
        bootstrap_q = parent_values.unsqueeze(-1).expand_as(child_values)  # [B, E]
    elif bootstrap_method == "zero":
        bootstrap_q = torch.zeros_like(child_values)
    elif bootstrap_method == "v_mix":
        from old_muzero.search.aos_search.scoring import compute_v_mix

        v_mix = compute_v_mix(tree, node_indices)
        bootstrap_q = v_mix.unsqueeze(-1).expand_as(child_values)
    elif bootstrap_method == "mu_fpu":
        sum_q_v = (child_values * child_visits).sum(dim=-1, keepdim=True)
        sum_v = child_visits.sum(dim=-1, keepdim=True)
        mu_fpu = torch.where(sum_v > 0, sum_q_v / sum_v, parent_values.unsqueeze(-1))
        bootstrap_q = mu_fpu.expand_as(child_values)
    else:
        bootstrap_q = parent_values.unsqueeze(-1).expand_as(child_values)  # [B, E]

    q_values = torch.where(visited_mask, child_values, bootstrap_q)  # [B, E]

    # --- Q normalisation ------------------------------------------------------
    # Use the explicit action mask to identify real (permitted) edges
    real_edge_mask = tree.children_action_mask[batch_idx, node_indices]  # [B, E]

    if min_max_stats is not None:
        value_score = min_max_stats.normalize(q_values.float())
    else:
        LARGE = 1e9
        q_for_min = torch.where(
            real_edge_mask,
            q_values,
            torch.tensor(LARGE, dtype=q_values.dtype, device=q_values.device),
        )
        q_min = q_for_min.min(dim=-1, keepdim=True).values

        q_for_max = torch.where(
            real_edge_mask,
            q_values,
            torch.tensor(-LARGE, dtype=q_values.dtype, device=q_values.device),
        )
        q_max = q_for_max.max(dim=-1, keepdim=True).values

        q_range = (q_max - q_min).clamp(min=1e-8)
        value_score = (q_values - q_min) / q_range

    # --- combine + mask padding -----------------------------------------------
    scores = prior_score + value_score
    scores = torch.where(
        real_edge_mask,
        scores,
        torch.tensor(-float("inf"), dtype=scores.dtype, device=scores.device),
    )
    return scores


# ---------------------------------------------------------------------------
# gumbel_score_fn  (Gumbel MuZero)
# ---------------------------------------------------------------------------


def gumbel_score_fn(
    tree: FlatTree,
    node_indices: torch.Tensor,
    gumbel_cvisit: float,
    gumbel_cscale: float,
    min_max_stats: Optional[VectorizedMinMaxStats] = None,
    bootstrap_method: str = "v_mix",
    gumbel_noise: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Compute Gumbel MuZero scores for every edge of the selected nodes.

    Replicates the vetted ``GumbelScoring.get_scores`` logic::

        completed_Q = Q(s,a) if visited, else v_mix(s)
        normalized_Q = min_max_normalize(completed_Q)
        σ(a) = (c_visit + max_N) * c_scale * normalized_Q(a)
        π_improved = softmax(log π₀ + σ)
        score(a) = π_improved(a) - N(s,a) / (1 + Σ_a N(s,a))

    Unvisited edges are bootstrapped using :func:`compute_v_mix` instead of
    the raw ``node_values``, exactly matching the original paper.

    Args:
        tree: A fully-allocated :class:`FlatTree` instance.
        node_indices: ``[B]`` int32 node indices to score.
        gumbel_cvisit: Gumbel ``c_visit`` constant.
        gumbel_cscale: Gumbel ``c_scale`` constant.
        min_max_stats: Optional global bounds for Q normalisation.

    Returns:
        Gumbel scores of shape ``[B, max_edges]``.
    """
    B = node_indices.shape[0]
    batch_idx = torch.arange(B, device=node_indices.device)

    # --- gather per-node stats ------------------------------------------------
    child_visits = tree.children_visits[batch_idx, node_indices].float()  # [B, E]
    child_values = tree.children_values[batch_idx, node_indices].float()  # [B, E]
    child_logits = tree.children_prior_logits[batch_idx, node_indices]  # [B, E]

    # Use the explicit action mask for edge validity
    real_edge_mask = tree.children_action_mask[batch_idx, node_indices]  # [B, E]
    expanded_mask = tree.children_index[batch_idx, node_indices] != -1
    visited_mask = expanded_mask

    # --- completed Q: visited → stored Q, unvisited → bootstrap ---------
    if bootstrap_method == "parent_value":
        bootstrap_q = (
            tree.node_values[batch_idx, node_indices]
            .unsqueeze(-1)
            .expand_as(child_values)
        )
    elif bootstrap_method == "zero":
        bootstrap_q = torch.zeros_like(child_values)
    elif bootstrap_method == "v_mix":
        v_mix = compute_v_mix(tree, node_indices)
        bootstrap_q = v_mix.unsqueeze(-1).expand_as(child_values)
    elif bootstrap_method == "mu_fpu":
        sum_q_v = (child_values * child_visits).sum(dim=-1, keepdim=True)
        sum_v = child_visits.sum(dim=-1, keepdim=True)
        parent_v = tree.node_values[batch_idx, node_indices].unsqueeze(-1)
        mu_fpu = torch.where(sum_v > 0, sum_q_v / sum_v, parent_v)
        bootstrap_q = mu_fpu.expand_as(child_values)
    else:
        v_mix = compute_v_mix(tree, node_indices)
        bootstrap_q = v_mix.unsqueeze(-1).expand_as(child_values)

    completed_q = torch.where(visited_mask, child_values, bootstrap_q)  # [B, E]

    # --- normalise completed Q ------------------------------------------------
    if min_max_stats is not None:
        norm_q = min_max_stats.normalize(completed_q)
    else:
        large = torch.tensor(1e9, dtype=completed_q.dtype, device=completed_q.device)
        q_for_min = torch.where(real_edge_mask, completed_q, large)
        q_for_max = torch.where(real_edge_mask, completed_q, -large)
        q_min = q_for_min.min(dim=-1, keepdim=True).values
        q_max = q_for_max.max(dim=-1, keepdim=True).values
        q_range = (q_max - q_min).clamp(min=1e-8)
        norm_q = (completed_q - q_min) / q_range

    # --- σ transformation -----------------------------------------------------
    max_n = child_visits.max(dim=-1, keepdim=True).values  # [B, 1]
    sigma = (gumbel_cvisit + max_n) * gumbel_cscale * norm_q  # [B, E]

    # --- improved policy: softmax(log π₀ + σ) ---------------------------------
    pi0_logits = child_logits + sigma  # [B, E]
    if gumbel_noise is not None:
        pi0_logits = pi0_logits + gumbel_noise
    # Mask padding before softmax
    pi0_logits = torch.where(
        real_edge_mask,
        pi0_logits,
        torch.full_like(pi0_logits, -float("inf")),
    )
    pi_improved = torch.softmax(pi0_logits, dim=-1)  # [B, E]

    # --- Gumbel score: π_improved - N / (1 + ΣN) -----------------------------
    sum_N = child_visits.sum(dim=-1, keepdim=True)  # [B, 1]
    visit_fraction = child_visits / (1.0 + sum_N)  # [B, E]
    scores = pi_improved - visit_fraction  # [B, E]

    # Kill padding
    scores = torch.where(
        real_edge_mask,
        scores,
        torch.tensor(-float("inf"), dtype=scores.dtype, device=scores.device),
    )
    return scores
