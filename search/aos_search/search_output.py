"""Pure, batched root-policy extraction functions for the FlatTree MCTS.

Replaces the OOP class hierarchy in ``search_py/root_policies.py``
(``VisitFrequencyPolicy``, ``CompletedQValuesRootPolicy``,
``BestActionRootPolicy``) with stateless functions that operate directly on
the ``FlatTree`` SoA.

All functions return a ``SearchOutput`` named-tuple containing:
  - ``target_policy``     — ``[B, num_actions]`` float32, training target
  - ``exploratory_policy``— ``[B, num_actions]`` float32, action selection
  - ``best_actions``      — ``[B]`` int64, greedy action per batch item
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from old_muzero.search.aos_search.tree import FlatTree


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------


class SearchOutput(NamedTuple):
    """Bundled output of every root-policy extractor."""

    target_policy: torch.Tensor  # [B, num_actions] float32
    exploratory_policy: torch.Tensor  # [B, num_actions] float32
    best_actions: torch.Tensor  # [B] int64
    root_values: torch.Tensor  # [B] float32


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Padding value in children_prior_logits / children_values for unused edges
_NEG_INF: float = -float("inf")


def _mask_padding(tensor: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
    """Zero-out slots whose prior logit is ``-inf`` (padding / unused edges).

    Args:
        tensor: ``[B, max_edges]`` float32 — values to mask.
        prior_logits: ``[B, max_edges]`` float32 — root prior logits.

    Returns:
        ``tensor`` with padding slots set to 0.
    """
    valid = prior_logits > _NEG_INF  # [B, max_edges]
    return torch.where(valid, tensor, torch.zeros_like(tensor))


def _normalize_rows(probs: torch.Tensor) -> torch.Tensor:
    """Row-wise normalise to a probability distribution.

    Args:
        probs: ``[B, max_edges]`` non-negative float32.

    Returns:
        ``[B, max_edges]`` with each row summing to 1.  Rows that sum to 0
        receive a uniform distribution (guarded by clamping).
    """
    row_sum = probs.sum(dim=-1, keepdim=True)  # [B, 1]
    return probs / row_sum.clamp(min=1e-8)


# ---------------------------------------------------------------------------
# visit_count_policy
# ---------------------------------------------------------------------------


def visit_count_policy(
    tree: FlatTree,
    num_actions: Optional[int] = None,
) -> SearchOutput:
    """AlphaZero-style policy proportional to root child visit counts.

    Args:
        tree: Populated :class:`FlatTree`.
        num_actions: If given, output is trimmed to ``num_actions`` columns.

    Returns:
        :class:`SearchOutput` where target equals exploratory (visit counts
        serve both roles in standard AlphaZero).
    """

    # Root is always node 0  →  children_visits[:, 0, :]
    # [B, max_edges] int32 → float
    visits = tree.children_visits[:, 0, :].float()
    prior_logits = tree.children_prior_logits[:, 0, :]  # [B, max_edges]

    # Mask padding slots (edges that were never assigned a prior)
    visits = _mask_padding(visits, prior_logits)

    if num_actions is not None:
        visits = visits[:, :num_actions]

    policy = _normalize_rows(visits)
    best_actions = policy.argmax(dim=-1)  # [B]

    return SearchOutput(
        target_policy=policy,
        exploratory_policy=policy,
        best_actions=best_actions,
        root_values=tree.node_values[:, 0],
    )


# ---------------------------------------------------------------------------
# gumbel_max_q_policy  (Gumbel MuZero improved policy)
# ---------------------------------------------------------------------------


def gumbel_max_q_policy(
    tree: FlatTree,
    gumbel_cvisit: float,
    gumbel_cscale: float,
    gumbel_noise: Optional[torch.Tensor] = None,
    min_max_stats=None,
    num_actions: Optional[int] = None,
) -> SearchOutput:
    """Gumbel MuZero policy extraction per Equation 11 / Algorithm 2.

    The two outputs use *different* logit combinations, as required by the
    paper::

        completed_Q   = Q(s,a)   if visited
                      = v_mix(s) if unvisited        (bias-free bootstrap)
        normalized_Q  = min_max_normalize(completed_Q)
        sigma(a)      = (c_visit + max_N) * c_scale * normalized_Q(a)

        # Equation 11 — training target (no Gumbel noise)
        improved_logits     = log pi_0(a) + sigma(a)
        target_policy       = softmax(improved_logits)   [masked, then softmaxed]

        # Algorithm 2 — action selection (Gumbel noise included)
        action_logits       = log pi_0(a) + g(a) + sigma(a)
        best_actions        = argmax(action_logits)
        exploratory_policy  = softmax(action_logits)

    Args:
        tree: Populated :class:`FlatTree`.
        gumbel_cvisit: Gumbel ``c_visit`` hyperparameter.
        gumbel_cscale: Gumbel ``c_scale`` hyperparameter.
        gumbel_noise: ``[B, num_actions]`` Gumbel noise sampled at root.  Pass
            ``None`` during evaluation (noise zeroed → deterministic argmax).
        min_max_stats: Optional :class:`VectorizedMinMaxStats` for global Q
            normalisation.  Falls back to per-batch local bounds when ``None``.
        num_actions: Trim output to this many action columns.

    Returns:
        :class:`SearchOutput` —
          * ``target_policy``     : ``softmax(log π₀ + σ)``  (training signal)
          * ``exploratory_policy``: ``softmax(log π₀ + g + σ)``
          * ``best_actions``      : ``argmax(log π₀ + g + σ)``
    """
    # [B, max_edges] — pristine log-probs (never clobbered by masking)
    prior_logits = tree.children_prior_logits[:, 0, :].clone()
    raw_visits = tree.children_visits[:, 0, :].float()  # [B, max_edges]
    q_values = tree.children_values[:, 0, :].float()  # [B, max_edges]

    if num_actions is not None:
        prior_logits = prior_logits[:, :num_actions]
        raw_visits = raw_visits[:, :num_actions]
        q_values = q_values[:, :num_actions]

    B, A = prior_logits.shape
    device = prior_logits.device

    # Action validity: use explicit mask, not logits>-inf (halving-safe)
    valid = tree.children_action_mask[:, 0, :A]  # [B, A] bool

    # --- completed_Q: visited -> stored Q, unvisited -> v_mix bootstrap ---
    visited = raw_visits > 0  # [B, A]

    # v_mix = (V(s) + sum_N * E_vis) / (1 + sum_N)  [B]
    child_priors_all = torch.softmax(prior_logits, dim=-1)  # [B, A]
    priors_vis = torch.where(
        visited, child_priors_all, torch.zeros_like(child_priors_all)
    )
    q_vis = torch.where(visited, q_values, torch.zeros_like(q_values))
    sum_priors_vis = priors_vis.sum(dim=-1).clamp(min=1e-8)  # [B]
    expected_q_vis = (priors_vis * q_vis).sum(dim=-1) / sum_priors_vis  # [B]
    sum_N = raw_visits.sum(dim=-1)  # [B]
    root_v = tree.raw_network_values[:, 0]  # [B]
    v_mix = (root_v + sum_N * expected_q_vis) / (1.0 + sum_N)  # [B]

    bootstrap = v_mix.unsqueeze(-1).expand(B, A)  # [B, A]
    completed_q = torch.where(visited, q_values, bootstrap)  # [B, A]

    # --- Normalize Q (global bounds when available, else local per-batch) ---
    if min_max_stats is not None:
        normalized_q = min_max_stats.normalize(completed_q)  # [B, A]
    else:
        large = torch.tensor(1e9, dtype=completed_q.dtype, device=device)
        q_for_min = torch.where(valid, completed_q, large)
        q_for_max = torch.where(valid, completed_q, -large)
        q_min = q_for_min.min(dim=-1, keepdim=True).values  # [B, 1]
        q_max = q_for_max.max(dim=-1, keepdim=True).values  # [B, 1]
        q_range = (q_max - q_min).clamp(min=1e-8)
        normalized_q = (completed_q - q_min) / q_range  # [B, A]

    # --- Sigma transformation ---
    max_n = raw_visits.max(dim=-1, keepdim=True).values  # [B, 1]
    sigma = (gumbel_cvisit + max_n) * gumbel_cscale * normalized_q  # [B, A]

    # --- Gumbel noise (zeros during eval / when not supplied) ---
    if gumbel_noise is None:
        noise = torch.zeros(B, A, dtype=prior_logits.dtype, device=device)
    else:
        assert gumbel_noise.shape == (B, A), (
            f"gumbel_noise shape mismatch: expected ({B}, {A}), "
            f"got {tuple(gumbel_noise.shape)}"
        )
        noise = gumbel_noise.to(dtype=prior_logits.dtype, device=device)

    # -----------------------------------------------------------------------
    # Target policy — Equation 11: softmax(log π₀ + σ)  [NO noise]
    # -----------------------------------------------------------------------
    improved_logits = prior_logits + sigma  # [B, A]
    improved_logits = torch.where(
        valid, improved_logits, torch.full_like(improved_logits, _NEG_INF)
    )
    target_policy = torch.softmax(improved_logits, dim=-1)  # [B, A]

    # -----------------------------------------------------------------------
    # Action logits — Algorithm 2: log π₀ + g + σ  [WITH noise]
    # -----------------------------------------------------------------------
    action_logits = prior_logits + noise + sigma  # [B, A]
    action_logits = torch.where(
        valid, action_logits, torch.full_like(action_logits, _NEG_INF)
    )

    # Best action (greedy on noisy logits)
    best_actions = action_logits.argmax(dim=-1)  # [B]

    exploratory_policy = torch.softmax(action_logits, dim=-1)  # [B, A]

    return SearchOutput(
        target_policy=target_policy,
        exploratory_policy=exploratory_policy,
        best_actions=best_actions,
        root_values=tree.node_values[:, 0],
    )


# ---------------------------------------------------------------------------
# minimax_policy
# ---------------------------------------------------------------------------


def minimax_policy(
    tree: FlatTree,
    num_actions: Optional[int] = None,
) -> SearchOutput:
    """Policy selecting the action with the highest Q-value at the root.

    The **target** policy is a one-hot over the argmax Q-value (used as the
    training signal).  The **exploratory** policy is a softmax over Q-values
    (for softer action selection during self-play).

    Args:
        tree: Populated :class:`FlatTree`.
        num_actions: Trim output to this many action columns.

    Returns:
        :class:`SearchOutput` with a one-hot target and soft exploratory policy.
    """
    # [B, max_edges]
    prior_logits = tree.children_prior_logits[:, 0, :]
    q_values = tree.children_values[:, 0, :].float()

    if num_actions is not None:
        prior_logits = prior_logits[:, :num_actions]
        q_values = q_values[:, :num_actions]

    valid = prior_logits > _NEG_INF  # [B, A]

    # Mask invalid (padding) slots to -inf
    q_masked = torch.where(valid, q_values, torch.full_like(q_values, _NEG_INF))

    # --- Target: one-hot argmax Q ---
    best_actions = q_masked.argmax(dim=-1)  # [B]
    B, A = q_values.shape
    target_policy = torch.zeros(B, A, device=q_values.device)
    target_policy.scatter_(1, best_actions.unsqueeze(1), 1.0)  # [B, A]

    # --- Exploratory: softmax over Q-values ---
    exploratory_policy = torch.softmax(q_masked, dim=-1)  # [B, A]

    return SearchOutput(
        target_policy=target_policy,
        exploratory_policy=exploratory_policy,
        best_actions=best_actions,
        root_values=tree.node_values[:, 0],
    )
