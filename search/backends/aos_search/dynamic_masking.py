"""Sequential Halving as flat-tensor dynamic masking for Gumbel MuZero.

Replaces the OOP ``SequentialHalvingPruning`` class.  Instead of maintaining
a per-batch Python list of survivors, we set
``tree.children_action_mask[:, 0, pruned]`` to ``False``, which permanently
prevents ``ucb_score_fn`` and ``batched_mcts_step`` from selecting those
actions in any subsequent simulation.  The raw
``children_prior_logits`` are preserved.

The core math follows the original Gumbel MuZero paper:
  * Budget per round: ``floor(N / (log2(m) * k)) * k``
  * Halving criterion: keep the top-``ceil(k/2)`` actions ranked by
    ``log π₀(a) + σ(a)`` where ``σ = (c_visit + max_N) * c_scale * Q̄(a)``.
"""

from __future__ import annotations

import math

import torch

from search.backends.aos_search.tree import FlatTree
from search.backends.aos_search.min_max_stats import VectorizedMinMaxStats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NEG_INF: float = -float("inf")

# Minimum number of survivors; never halve below this.
MIN_SURVIVORS: int = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_sequential_halving(
    tree: FlatTree,
    current_sim_idx: int,
    total_simulations: int,
    base_m: int,
    gumbel_cvisit: float,
    gumbel_cscale: float,
    gumbel_noise: torch.Tensor,
    min_max_stats: VectorizedMinMaxStats | None = None,
    num_actions: int | None = None,
) -> None:
    """Conditionally prune root actions via Sequential Halving.

    Call this at the **top** of every simulation iteration.  The function is
    a no-op when the current simulation index does not trigger a halving
    phase.

    When a phase *is* triggered the function:
      1.  Counts how many actions are still alive (logits > -inf) per batch
          item.
      2.  Halves those survivors down to ``ceil(k/2)`` (minimum 2).
      3.  Ranks survivors by the Gumbel scoring rule (Algorithm 2)
          ``log π₀(a) + g(a) + σ(a)`` where ``g`` is the Gumbel noise
          sampled at the root and
          ``σ = (c_visit + max_N) * c_scale * normalized_Q(a)``.
      4.  Sets ``children_action_mask[:, 0, pruned]`` to ``False``.

    Args:
        tree: :class:`FlatTree` to mutate in-place (root = node 0).
        current_sim_idx: Zero-based index of the simulation about to run.
        total_simulations: ``config.num_simulations``.
        base_m: Initial number of candidate actions (``config.gumbel_m``
            or the number of initially selected actions).
        gumbel_cvisit: Gumbel ``c_visit`` hyperparameter.
        gumbel_cscale: Gumbel ``c_scale`` hyperparameter.
        gumbel_noise: ``[B, A]`` float32 Gumbel noise sampled at the root
            before simulations began.  Must be the same tensor used for the
            final policy extraction (``gumbel_max_q_policy``).
        min_max_stats: Optional global bounds for Q normalisation.  When
            ``None``, a local min-max across root Q-values is used.
        num_actions: Number of game actions (columns to consider).  Defaults
            to ``max_edges`` of the tree.
    """
    if base_m <= MIN_SURVIVORS:
        return  # nothing to halve

    device = tree.node_visits.device

    # --- Determine the halving schedule ------------------------------------
    # phase_boundaries[i] = cumulative budget through phase i.
    # Budget for phase with k survivors:
    #   sims_per_survivor = max(1, floor(N / (log2(m) * k)))
    #   budget = sims_per_survivor * k
    phase_boundary = _get_phase_boundary(current_sim_idx, total_simulations, base_m)
    if phase_boundary is None:
        return  # not a halving point

    # --- Identify currently alive actions at the root ----------------------
    max_edges = tree.children_prior_logits.shape[-1]
    A = num_actions if num_actions is not None else max_edges

    # [B, A] — alive = still permitted by the action mask
    alive = tree.children_action_mask[:, 0, :A]  # [B, A] bool

    # Read pristine logits (never overwritten)
    root_logits = tree.children_prior_logits[:, 0, :A]  # [B, A]

    # Count alive per batch item
    # [B]
    num_alive = alive.sum(dim=-1)

    # Target number of survivors after this halving
    # [B] int — ceil(k / 2), clamp to MIN_SURVIVORS
    new_k = (num_alive.float() / 2.0).ceil().clamp(min=MIN_SURVIVORS).int()  # [B]

    # --- Rank survivors by Gumbel score: log π₀ + σ -----------------------
    # Q-values and visits at the root
    root_q = tree.children_values[:, 0, :A].float()  # [B, A]
    root_visits = tree.children_visits[:, 0, :A].float()  # [B, A]

    # Normalise Q-values
    if min_max_stats is not None:
        norm_q = min_max_stats.normalize(root_q)  # [B, A]
    else:
        # Local per-batch min-max
        large = torch.tensor(1e9, dtype=root_q.dtype, device=device)
        q_for_min = torch.where(alive, root_q, large)
        q_for_max = torch.where(alive, root_q, -large)
        q_min = q_for_min.min(dim=-1, keepdim=True).values  # [B, 1]
        q_max = q_for_max.max(dim=-1, keepdim=True).values  # [B, 1]
        rng = (q_max - q_min).clamp(min=1e-8)
        norm_q = (root_q - q_min) / rng  # [B, A]

    # σ = (c_visit + max_N) * c_scale * normalized_Q
    max_n = root_visits.max(dim=-1, keepdim=True).values  # [B, 1]
    sigma = (gumbel_cvisit + max_n) * gumbel_cscale * norm_q  # [B, A]

    # score = log π₀ + g + σ  (Algorithm 2 from Gumbel MuZero paper)
    assert (
        gumbel_noise.shape[0] == B and gumbel_noise.shape[1] >= A
    ), f"gumbel_noise must be [B, >=A], got {tuple(gumbel_noise.shape)}"
    noise_at_root = gumbel_noise[:, :A].to(dtype=root_logits.dtype, device=device)
    scores = root_logits + noise_at_root + sigma  # [B, A]

    # Dead actions must not win the ranking
    scores = torch.where(alive, scores, torch.full_like(scores, _NEG_INF))

    # --- Keep top new_k per batch item ------------------------------------
    # Sort descending
    _, sorted_indices = scores.sort(dim=-1, descending=True)  # [B, A]

    # Build a keep-mask:  keep[b, sorted_indices[b, j]] = True iff j < new_k[b]
    B = root_logits.shape[0]
    rank_positions = torch.arange(A, device=device).unsqueeze(0).expand(B, -1)  # [B, A]
    keep_by_rank = rank_positions < new_k.unsqueeze(-1)  # [B, A] bool

    # Scatter back to original action indices
    keep_mask = torch.zeros(B, A, dtype=torch.bool, device=device)
    keep_mask.scatter_(1, sorted_indices, keep_by_rank)

    # --- Prune: set non-kept entries in the action mask to False -----------
    tree.children_action_mask[:, 0, :A] = keep_mask


# ---------------------------------------------------------------------------
# Budget maths
# ---------------------------------------------------------------------------


def _get_phase_boundary(
    current_sim_idx: int,
    total_simulations: int,
    base_m: int,
) -> int | None:
    """Return the current phase boundary index if we just hit one, else None.

    Phase boundaries are the cumulative budget endpoints for each halving
    round.  If ``current_sim_idx`` equals a boundary, a halving is triggered.
    """
    if base_m <= MIN_SURVIVORS:
        return None

    log_m = max(1.0, math.log2(base_m))
    k = base_m
    cumulative = 0

    while k > MIN_SURVIVORS:
        sims_per_survivor = max(1, math.floor(total_simulations / (log_m * k)))
        budget = sims_per_survivor * k
        cumulative += budget

        # Clamp to total budget
        if cumulative > total_simulations:
            cumulative = total_simulations

        if current_sim_idx == cumulative:
            return cumulative

        k = max(MIN_SURVIVORS, math.ceil(k / 2))

    return None
