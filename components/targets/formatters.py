"""
Pure projection components for the learner target pipeline.

Design contract (per philosophy.md):
  - Components are stateless transforms: they READ from the blackboard and WRITE back.
  - No network calls, no optimiser state, no side-effects.
  - All tensor operations are vectorised; no Python loops over batch dimension.

Blackboard key conventions:
  - raw_values       : [B, T] scalar targets (raw bootstrap / MCTS values).
  - projected_values : [B, T, bins] two-hot distributions ready for cross-entropy loss.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from core import PipelineComponent, Blackboard
from modules.representations import BaseRepresentation, DiscreteSupportRepresentation

if TYPE_CHECKING:
    pass  # guard only used for type annotations if needed in future

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RAW_VALUES_KEY: str = "raw_values"
_PROJECTED_VALUES_KEY: str = "projected_values"


# ---------------------------------------------------------------------------
# TwoHotProjectionComponent
# ---------------------------------------------------------------------------


class TwoHotProjectionComponent(PipelineComponent):
    """Projects scalar targets onto a discrete support via two-hot encoding.

    Reads  ``blackboard.targets['raw_values']`` (shape ``[B, T]`` or ``[B]``)
    and writes ``blackboard.targets['projected_values']`` (shape ``[B, T, bins]``).

    This is the canonical two-hot (categorical-support) projection used by
    MuZero-style algorithms.  All maths are delegated to
    :class:`~learner.losses.representations.DiscreteSupportRepresentation` so
    that the support geometry is defined in exactly one place.

    Args:
        representation: A :class:`~learner.losses.representations.DiscreteSupportRepresentation`
            (or any subclass) that defines ``vmin``, ``vmax``, and ``bins``.
        source_key: Blackboard key to read scalar targets from (default
            ``"raw_values"``).
        dest_key: Blackboard key to write the projected distribution to (default
            ``"projected_values"``).
    """

    def __init__(
        self,
        representation: DiscreteSupportRepresentation,
        source_key: str = _RAW_VALUES_KEY,
        dest_key: str = _PROJECTED_VALUES_KEY,
    ) -> None:
        assert isinstance(representation, DiscreteSupportRepresentation), (
            f"TwoHotProjectionComponent requires a DiscreteSupportRepresentation, "
            f"got {type(representation).__name__}"
        )
        self._representation = representation
        self._source_key = source_key
        self._dest_key = dest_key

    def execute(self, blackboard: Blackboard) -> None:
        """Project scalar targets to two-hot distributions and write back.

        Args:
            blackboard: The shared pipeline blackboard.  The component will
                read ``targets[source_key]`` and write
                ``targets[dest_key]``.

        Raises:
            AssertionError: If ``source_key`` is missing from
                ``blackboard.targets``.
        """
        assert self._source_key in blackboard.targets, (
            f"TwoHotProjectionComponent: expected '{self._source_key}' in "
            f"blackboard.targets, but found keys: {list(blackboard.targets.keys())}"
        )

        raw: torch.Tensor = blackboard.targets[self._source_key]  # [B, T] or [B]

        # Delegate all projection maths to the representation.
        # to_representation handles arbitrary leading dims and returns [..., bins].
        projected: torch.Tensor = self._representation.to_representation(raw)
        # projected: [B, T, bins] (or [B, bins] if raw was 1-D)

        blackboard.targets[self._dest_key] = projected


# ---------------------------------------------------------------------------
# ExpectedValueComponent
# ---------------------------------------------------------------------------


class ExpectedValueComponent(PipelineComponent):
    """Collapses categorical logits into a scalar expected value.

    Reads raw logits from ``blackboard.predictions[logits_key]``
    (shape ``[B, T, bins]`` or ``[B, bins]``) and writes a scalar tensor
    to ``blackboard.targets[dest_key]`` (shape ``[B, T]`` or ``[B]``).

    This component is useful when a downstream loss (e.g. MSE) needs a scalar
    prediction of the value head rather than the full categorical distribution.

    Args:
        representation: Any :class:`~learner.losses.representations.BaseRepresentation`
            that implements ``to_expected_value``.
        logits_key: Key in ``blackboard.predictions`` where logits are stored.
        dest_key: Key in ``blackboard.targets`` where the scalar will be written.
    """

    def __init__(
        self,
        representation: BaseRepresentation,
        logits_key: str,
        dest_key: str,
    ) -> None:
        assert isinstance(representation, BaseRepresentation), (
            f"ExpectedValueComponent requires a BaseRepresentation, "
            f"got {type(representation).__name__}"
        )
        self._representation = representation
        self._logits_key = logits_key
        self._dest_key = dest_key

    def execute(self, blackboard: Blackboard) -> None:
        """Compute expected value from logits and write to targets.

        Args:
            blackboard: The shared pipeline blackboard.  The component reads
                ``predictions[logits_key]`` and writes
                ``targets[dest_key]``.

        Raises:
            AssertionError: If ``logits_key`` is missing from
                ``blackboard.predictions``.
        """
        assert self._logits_key in blackboard.predictions, (
            f"ExpectedValueComponent: expected '{self._logits_key}' in "
            f"blackboard.predictions, but found keys: "
            f"{list(blackboard.predictions.keys())}"
        )

        logits: torch.Tensor = blackboard.predictions[self._logits_key]
        # Shape: [B, T, bins] or [B, bins] — to_expected_value handles both.
        scalar: torch.Tensor = self._representation.to_expected_value(logits)
        # Shape: [B, T] or [B]

        blackboard.targets[self._dest_key] = scalar
