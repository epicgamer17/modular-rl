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
from typing import TYPE_CHECKING, Optional

from core import PipelineComponent, Blackboard
from core.path_resolver import resolve_blackboard_path
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

    Reads from a source (path or key) and writes a [B, T, bins] distribution to 
    ``blackboard.targets[dest_key]``.

    This is the canonical two-hot (categorical-support) projection used by
    MuZero-style algorithms. All maths are delegated to
    :class:`~learner.losses.representations.DiscreteSupportRepresentation`.

    Args:
        source_key: Blackboard key or path to read scalar targets from.
        dest_key: Blackboard key to write the projected distribution to.
        representation: Optional discrete support representation.
        v_min: Optional min value for support (used if representation is None).
        v_max: Optional max value for support (used if representation is None).
        bins: Optional number of bins for support (used if representation is None).
    """

    def __init__(
        self,
        source_key: str,
        dest_key: str,
        representation: Optional[DiscreteSupportRepresentation] = None,
        v_min: Optional[float] = None,
        v_max: Optional[float] = None,
        bins: Optional[int] = None,
    ) -> None:
        if representation is None:
            assert v_min is not None and v_max is not None and bins is not None, (
                "TwoHotProjectionComponent requires either a representation or v_min, v_max, and bins."
            )
            representation = DiscreteSupportRepresentation(v_min, v_max, bins)
        
        assert isinstance(representation, DiscreteSupportRepresentation), (
            f"TwoHotProjectionComponent requires a DiscreteSupportRepresentation, "
            f"got {type(representation).__name__}"
        )
        self._representation = representation
        self._source_key = source_key
        self._dest_key = dest_key

    def execute(self, blackboard: Blackboard) -> None:
        """Project scalar targets to two-hot distributions and write back."""
        raw = resolve_blackboard_path(blackboard, self._source_key)
        
        # projected: [B, T, bins] (or [B, bins] if raw was 1-D)
        projected: torch.Tensor = self._representation.to_representation(raw)

        blackboard.targets[self._dest_key] = projected


# ---------------------------------------------------------------------------
# ClassificationFormatterComponent
# ---------------------------------------------------------------------------


class ClassificationFormatterComponent(PipelineComponent):
    """Formats classification targets (e.g. policies).

    If the target is already a distribution (ndim > 1), it passes through.
    Otherwise, it converts class indices into one-hot vectors.

    Args:
        source_key: Blackboard key or path to read targets from.
        dest_key: Blackboard key to write final targets to.
        representation: Optional ClassificationRepresentation.
    """

    def __init__(
        self,
        source_key: str,
        dest_key: str,
        representation: Optional[BaseRepresentation] = None,
    ) -> None:
        self._source_key = source_key
        self._dest_key = dest_key
        self._representation = representation

    def execute(self, blackboard: Blackboard) -> None:
        val = resolve_blackboard_path(blackboard, self._source_key)
        
        if self._representation is not None:
            formatted = self._representation.format_target(
                {self._dest_key: val}, target_key=self._dest_key
            )
        else:
            # Simple identity fallback if no representation is provided
            formatted = val
            
        blackboard.targets[self._dest_key] = formatted


# ---------------------------------------------------------------------------
# ScalarFormatterComponent
# ---------------------------------------------------------------------------


class ScalarFormatterComponent(PipelineComponent):
    """Formats scalar targets (e.g. rewards, to-play).

    Useful for ensuring targets have the correct canonical shape [B, T, 1] 
    and are placed in the targets container.

    Args:
        source_key: Blackboard key or path to read targets from.
        dest_key: Blackboard key to write final targets to.
        representation: Optional ScalarRepresentation.
    """

    def __init__(
        self,
        source_key: str,
        dest_key: str,
        representation: Optional[BaseRepresentation] = None,
    ) -> None:
        self._source_key = source_key
        self._dest_key = dest_key
        self._representation = representation

    def execute(self, blackboard: Blackboard) -> None:
        val = resolve_blackboard_path(blackboard, self._source_key)
        
        if self._representation is not None:
            formatted = self._representation.format_target(
                {self._dest_key: val}, target_key=self._dest_key
            )
        else:
            # Canonical [B, T] -> [B, T, 1] if it's not already
            if torch.is_tensor(val) and val.ndim == 2:
                formatted = val.unsqueeze(-1)
            else:
                formatted = val

        blackboard.targets[self._dest_key] = formatted


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
        """Compute expected value from logits and write to targets."""
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
