"""Vectorized, batched min-max bounds tracker for MCTS Q-value normalisation.

Replaces the scalar OOP ``MinMaxStats`` in ``search_py/min_max_stats.py``
with a dataclass that tracks bounds **per batch element** as GPU tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Sentinels used to detect "no values seen yet"
_POS_INF: float = float("inf")
_NEG_INF: float = -float("inf")


@dataclass
class VectorizedMinMaxStats:
    """Per-batch min/max bounds for Q-value normalisation.

    Maintains running ``min_values`` and ``max_values`` tensors of shape
    ``[B]``.  After each backpropagation step, call :meth:`update` with the
    new Q-values so the bounds grow to accommodate them.  During scoring,
    call :meth:`normalize` to map any ``[B, A]`` Q-tensor into ``[0, 1]``.

    Attributes:
        min_values: ``[B]`` float32 – running per-batch minima,
            initialised to ``+inf`` (no values seen yet).
        max_values: ``[B]`` float32 – running per-batch maxima,
            initialised to ``-inf`` (no values seen yet).
    """

    min_values: torch.Tensor  # [B] float32
    max_values: torch.Tensor  # [B] float32
    epsilon: float = 1e-8

    @classmethod
    def allocate(
        cls,
        batch_size: int,
        device: torch.device,
        known_bounds: list[float] | None = None,
        epsilon: float = 1e-8,
    ) -> "VectorizedMinMaxStats":
        """Allocate fresh, uninitialised bounds (no values seen).

        Args:
            batch_size: Number of parallel search instances ``B``.
            device: Torch device for the tensors.
            known_bounds: Optional `[min, max]` list of floats to initialize the
                min and max values with. If a bound is None, it defaults to INF.
            epsilon: Small constant for numerical stability (soft min-max).

        Returns:
            :class:`VectorizedMinMaxStats` with initialized fields.
        """
        init_min = _POS_INF
        init_max = _NEG_INF

        if known_bounds is not None:
            if known_bounds[0] is not None:
                init_min = float(known_bounds[0])
            if known_bounds[1] is not None:
                init_max = float(known_bounds[1])

        return cls(
            min_values=torch.full(
                (batch_size,), init_min, dtype=torch.float32, device=device
            ),
            max_values=torch.full(
                (batch_size,), init_max, dtype=torch.float32, device=device
            ),
            epsilon=epsilon,
        )

    def update(self, new_q_values: torch.Tensor, valid_mask: torch.Tensor) -> None:
        """Expand per-batch bounds to include new Q-values.

        Args:
            new_q_values: ``[B, A]`` or ``[B]`` float32 — Q-values observed
                at this step.
            valid_mask: Boolean tensor broadcastable to ``new_q_values``
                — only expand bounds where ``True``.
        """
        assert (
            new_q_values.dtype == torch.float32
        ), f"new_q_values must be float32, got {new_q_values.dtype}"
        assert (
            valid_mask.dtype == torch.bool
        ), f"valid_mask must be bool, got {valid_mask.dtype}"

        device = new_q_values.device

        if new_q_values.dim() == 2:
            # [B, A] — reduce to per-batch extrema, ignoring invalid slots
            pos = torch.tensor(_POS_INF, dtype=torch.float32, device=device)
            neg = torch.tensor(_NEG_INF, dtype=torch.float32, device=device)
            masked_for_min = torch.where(valid_mask, new_q_values, pos)
            masked_for_max = torch.where(valid_mask, new_q_values, neg)
            batch_min = masked_for_min.min(dim=-1).values  # [B]
            batch_max = masked_for_max.max(dim=-1).values  # [B]
        else:
            # [B] — one Q-value per batch item
            assert (
                new_q_values.dim() == 1
            ), f"Expected 1-D or 2-D new_q_values, got shape {new_q_values.shape}"
            pos_1d = torch.tensor(_POS_INF, dtype=torch.float32, device=device)
            neg_1d = torch.tensor(_NEG_INF, dtype=torch.float32, device=device)
            batch_min = torch.where(valid_mask, new_q_values, pos_1d)
            batch_max = torch.where(valid_mask, new_q_values, neg_1d)

        # Expand running min/max elementwise (no Python loops)
        self.min_values = torch.minimum(self.min_values, batch_min)
        self.max_values = torch.maximum(self.max_values, batch_max)

    def normalize(self, q_values: torch.Tensor) -> torch.Tensor:
        """Normalise Q-values to ``[0, 1]`` using per-batch bounds.

        When bounds are uninitialised (range == 0 or bounds are ±inf) the
        raw ``q_values`` are returned unchanged — matching the behaviour of
        the scalar ``MinMaxStats`` class.

        Args:
            q_values: ``[B, A]`` or ``[B]`` float32 tensor to normalise.

        Returns:
            Normalised tensor of the same shape.
        """
        assert (
            q_values.dtype == torch.float32
        ), f"q_values must be float32, got {q_values.dtype}"

        if q_values.dim() == 2:
            lo = self.min_values.unsqueeze(-1)  # [B, 1] — broadcasts over A
            hi = self.max_values.unsqueeze(-1)  # [B, 1]
        else:
            lo = self.min_values  # [B]
            hi = self.max_values  # [B]

        delta = hi - lo  # [B, 1] or [B]

        # Branchless normalization (matches mctx behaviour):
        # when min == max, q_values == lo, so numerator == 0 -> 0 / eps = 0.0.
        # No conditional needed; clamping the denominator is sufficient.
        # The final clamp(0.0, 1.0) strictly bounds everything (e.g. v_mix bootstrap).
        return ((q_values - lo) / delta.clamp(min=self.epsilon)).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# StaticMinMaxStats
# ---------------------------------------------------------------------------


@dataclass
class StaticMinMaxStats:
    """Fixed-bound Q-value normaliser for games with known score ranges.

    Unlike :class:`VectorizedMinMaxStats`, the bounds are set at construction
    time and never change.  ``update`` is a no-op.  This is the correct choice
    for environments where the theoretical min/max return is known in advance
    (e.g. board games with ``[-1, 1]`` outcomes), as it avoids the instability
    that can arise when the running bounds are estimated from early, noisy
    samples.

    The ``update`` / ``normalize`` interface is identical to
    :class:`VectorizedMinMaxStats`, so ``StaticMinMaxStats`` can be passed
    anywhere that accepts ``VectorizedMinMaxStats``.

    Attributes:
        minimum: Hard lower bound on Q-values.
        maximum: Hard upper bound on Q-values.

    Example::

        stats = StaticMinMaxStats(minimum=-1.0, maximum=1.0)
        normalized = stats.normalize(q_tensor)   # always in [0, 1]
    """

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        assert self.minimum < self.maximum, (
            f"minimum ({self.minimum}) must be strictly less than "
            f"maximum ({self.maximum})"
        )

    def update(
        self, new_q_values: torch.Tensor, valid_mask: torch.Tensor
    ) -> None:  # noqa: ARG002
        """No-op — bounds are statically fixed and never change.

        Args:
            new_q_values: Ignored.
            valid_mask: Ignored.
        """
        return

    def normalize(self, q_values: torch.Tensor) -> torch.Tensor:
        """Map ``q_values`` to ``[0, 1]`` using the fixed bounds.

        Args:
            q_values: Any-shape float32 tensor of Q-values.

        Returns:
            Tensor of the same shape with values in ``[0, 1]``.
        """
        assert (
            q_values.dtype == torch.float32
        ), f"q_values must be float32, got {q_values.dtype}"
        q_range = max(self.maximum - self.minimum, 1e-8)
        return ((q_values - self.minimum) / q_range).clamp(0.0, 1.0)

