"""Pure functional modifiers that operate on batched logit tensors.

These replace the OOP class hierarchies in search_py/prior_injectors.py,
search_py/initial_searchsets.py, and search_py/pruners.py with vectorized
tensor operations that run entirely on the GPU.

All functions expect batched inputs (leading batch dimension ``B``) and
use **no** Python ``for`` loops.
"""

import torch


def apply_dirichlet_noise(
    logits: torch.Tensor,
    alpha: float,
    fraction: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Blend Dirichlet noise into softmaxed logits and return new log-probs.

    When ``valid_mask`` is supplied, the noise is restricted to legal actions
    only — illegal actions receive 0 probability both before and after blending.
    The blending formula is::

        p_legal = softmax(logits)        with illegal actions zeroed out
        noise   = Dir(alpha)             then zeroed on illegal slots and
                  re-normalised so Σ_legal noise == 1.0
        p_noisy = (1 - fraction) * p_legal + fraction * noise
        return log(p_noisy)              with 0-probability slots → -inf

    Args:
        logits: Raw policy logits, shape ``[B, num_actions]``.
        alpha: Concentration parameter for the symmetric Dirichlet
            distribution.  A common heuristic is ``1 / sqrt(num_legal)``.
        fraction: Exploration fraction in ``[0, 1]``.
        valid_mask: Optional ``[B, num_actions]`` bool tensor.  ``True`` marks
            legal actions.  When ``None``, all actions are treated as legal
            (original behaviour).

    Returns:
        Log-probabilities with injected noise, shape ``[B, num_actions]``.
    """
    assert (
        logits.dim() == 2
    ), f"Expected batched logits [B, num_actions], got {logits.shape}"
    assert 0.0 <= fraction <= 1.0, f"fraction must be in [0, 1], got {fraction}"

    B, A = logits.shape
    device = logits.device
    dtype = logits.dtype

    # --- Build effective mask (all-True when not supplied) ---
    if valid_mask is not None:
        assert valid_mask.shape == (B, A), (
            f"valid_mask shape mismatch: expected ({B}, {A}), "
            f"got {tuple(valid_mask.shape)}"
        )
        mask = valid_mask.float()  # [B, A] 1.0 / 0.0
    else:
        mask = torch.ones(B, A, dtype=dtype, device=device)

    # --- Masked softmax: probs only over legal actions ---
    masked_logits = torch.where(
        mask.bool(),
        logits,
        torch.full_like(logits, -float("inf")),
    )
    probs = torch.softmax(masked_logits, dim=-1)  # [B, A]

    # --- Dirichlet noise sampled over the full action space, then masked ---
    concentration = torch.full((A,), alpha, dtype=dtype, device=device)
    dirichlet = torch.distributions.Dirichlet(concentration)
    noise = dirichlet.sample((B,))  # [B, A]

    # Zero out illegal actions and re-normalise so Σ_legal noise == 1.0
    noise = noise * mask  # [B, A]
    noise_sum = noise.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, 1]
    noise = noise / noise_sum  # [B, A]

    # --- Blend ---
    blended = (1.0 - fraction) * probs + fraction * noise  # [B, A]

    # --- Safe log: 0-probability slots become -inf ---
    log_prob = torch.where(
        blended > 0,
        blended.log(),
        torch.full_like(blended, -float("inf")),
    )

    return log_prob


def mask_invalid_actions(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Mask invalid actions by setting their logits to ``-inf``.

    Args:
        logits: Raw policy logits, shape ``[B, num_actions]``.
        valid_mask: Boolean tensor, shape ``[B, num_actions]``.
            ``True`` marks a **valid** (legal) action.

    Returns:
        Masked logits with the same shape, where invalid entries are
        ``-inf``.
    """
    assert (
        logits.shape == valid_mask.shape
    ), f"Shape mismatch: logits {logits.shape} vs valid_mask {valid_mask.shape}"
    assert (
        valid_mask.dtype == torch.bool
    ), f"valid_mask must be bool, got {valid_mask.dtype}"

    # [B, num_actions] → invalid positions become -inf
    return torch.where(
        valid_mask,
        logits,
        torch.tensor(-float("inf"), dtype=logits.dtype, device=logits.device),
    )


def force_trajectory_actions(
    logits: torch.Tensor,
    trajectory_actions: torch.Tensor,
) -> torch.Tensor:
    """Force specific actions by masking all others to ``-inf``.

    For each batch item where ``trajectory_actions[b] >= 0``, every action
    **except** that index is set to ``-inf``, guaranteeing that MCTS will
    only explore the forced action on its first selection step.  Batch items
    with ``trajectory_actions[b] == -1`` are left unmodified.

    This replaces the OOP ``ActionTargetInjector`` / ``InitialSearchset``
    pattern entirely.

    Args:
        logits: Policy logits, shape ``[B, num_actions]``.
        trajectory_actions: ``[B]`` int64 — action index to force, or
            ``-1`` for "no forced action".

    Returns:
        Modified logits with the same shape.
    """
    assert (
        logits.dim() == 2
    ), f"Expected batched logits [B, num_actions], got {logits.shape}"
    assert trajectory_actions.shape[0] == logits.shape[0], (
        f"Batch size mismatch: logits {logits.shape[0]} vs "
        f"trajectory_actions {trajectory_actions.shape[0]}"
    )

    B, A = logits.shape

    # [B] bool — which batch items have a forced action
    has_forced = trajectory_actions >= 0  # [B]

    if not has_forced.any():
        return logits

    # Build a [B, A] mask: True only at the forced action index
    # scatter_ a True into the correct column for each forced batch item
    forced_mask = torch.zeros(B, A, dtype=torch.bool, device=logits.device)
    # Clamp to 0 for the scatter (items with -1 won't be used anyway)
    safe_actions = trajectory_actions.clamp(min=0).unsqueeze(-1)  # [B, 1]
    forced_mask.scatter_(1, safe_actions, True)  # [B, A]

    # For forced items: keep only the forced action, -inf everything else
    # For non-forced items: keep original logits
    # [B, 1] broadcast
    is_forced_row = has_forced.unsqueeze(-1)  # [B, 1]

    result = torch.where(
        is_forced_row & ~forced_mask,
        torch.tensor(-float("inf"), dtype=logits.dtype, device=logits.device),
        logits,
    )

    return result
