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
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Set
from core.contracts import (
    Key,
    ShapeContract,
    ValueTarget,
    Policy,
    Action,
    Reward,
    Scalar,
    Logits,
    Probs,
    LogProbs,
    Categorical,
    SemanticType,
)
from modules.representations import (
    BaseRepresentation,
    DiscreteSupportRepresentation,
    ClassificationRepresentation,
)
from core import PipelineComponent, Blackboard
from core.path_resolver import resolve_blackboard_path

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
        semantic_type: Type[SemanticType] = SemanticType,
    ) -> None:
        if representation is None:
            assert (
                v_min is not None and v_max is not None and bins is not None
            ), "TwoHotProjectionComponent requires either a representation or v_min, v_max, and bins."
            representation = DiscreteSupportRepresentation(v_min, v_max, bins)

        assert isinstance(representation, DiscreteSupportRepresentation), (
            f"TwoHotProjectionComponent requires a DiscreteSupportRepresentation, "
            f"got {type(representation).__name__}"
        )
        self._representation = representation
        self._source_key = source_key
        self._dest_key = dest_key
        self._semantic_type = semantic_type

        # Deterministic contracts
        struct = self._representation.get_structure()
        metadata = self._representation.get_metadata()
        self._requires = {Key(self._source_key, self._semantic_type[Scalar])}
        self._provides = {
            Key(
                f"targets.{self._dest_key}",
                self._semantic_type[struct],
                metadata=metadata,
            ): "new"
        }

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures source key exists and contains a scalar tensor."""
        from core.validation import (
            assert_in_blackboard,
            assert_is_tensor,
            assert_shape_sanity,
        )

        assert_in_blackboard(blackboard, self._source_key)

        raw = resolve_blackboard_path(blackboard, self._source_key)
        assert_is_tensor(raw, msg=f"for {self.__class__.__name__} ({self._source_key})")
        # Two-hot usually expects [B] or [B, T]
        assert_shape_sanity(
            raw, min_ndim=1, max_ndim=2, msg=f"for {self.__class__.__name__}"
        )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """Project scalar targets to two-hot distributions and write back."""
        raw = resolve_blackboard_path(blackboard, self._source_key)

        # Use format_target for built-in validation of ingredients and output
        projected = self._representation.format_target(
            {self._dest_key: raw}, target_key=self._dest_key
        )

        return {f"targets.{self._dest_key}": projected}


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
        semantic_type: Type[SemanticType] = SemanticType,
    ) -> None:
        self._source_key = source_key
        self._dest_key = dest_key
        self._representation = representation
        self._semantic_type = semantic_type

        # Deterministic contracts
        struct = Scalar()
        metadata = {}
        if self._representation is not None:
            struct = self._representation.get_structure()
            metadata = self._representation.get_metadata()

        self._requires = {Key(self._source_key, self._semantic_type)}
        self._provides = {
            Key(
                f"targets.{self._dest_key}",
                self._semantic_type[struct],
                metadata=metadata,
            ): "new"
        }

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures source exists and is a tensor."""
        from core.validation import assert_in_blackboard, assert_is_tensor

        assert_in_blackboard(blackboard, self._source_key)
        val = resolve_blackboard_path(blackboard, self._source_key)
        assert_is_tensor(val, msg=f"for {self.__class__.__name__} ({self._source_key})")

        # If representation exists, it might have its own validation logic
        if self._representation is not None and hasattr(
            self._representation, "validate_logits"
        ):
            self._representation.validate_logits(val)

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        val = resolve_blackboard_path(blackboard, self._source_key)

        if self._representation is not None:
            formatted = self._representation.format_target(
                {self._dest_key: val}, target_key=self._dest_key
            )
        else:
            # Simple identity fallback if no representation is provided
            formatted = val

        return {f"targets.{self._dest_key}": formatted}


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
        semantic_type: Type[SemanticType] = SemanticType,
    ) -> None:
        self._source_key = source_key
        self._dest_key = dest_key
        self._representation = representation
        self._semantic_type = semantic_type

        # Deterministic contracts
        struct = Scalar()
        metadata = {}
        if self._representation is not None:
            struct = self._representation.get_structure()
            metadata = self._representation.get_metadata()

        self._requires = {Key(self._source_key, self._semantic_type[Scalar])}
        self._provides = {
            Key(
                f"targets.{self._dest_key}",
                self._semantic_type[struct],
                metadata=metadata,
            ): "new"
        }

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures source exists and is a tensor."""
        from core.validation import assert_in_blackboard, assert_is_tensor

        assert_in_blackboard(blackboard, self._source_key)
        val = resolve_blackboard_path(blackboard, self._source_key)
        assert_is_tensor(val, msg=f"for {self.__class__.__name__} ({self._source_key})")

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
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

        return {f"targets.{self._dest_key}": formatted}


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

        # Deterministic contracts
        struct = self._representation.get_structure()
        metadata = self._representation.get_metadata()

        self._requires = {
            Key(
                f"predictions.{self._logits_key}",
                Policy[Probs],
                metadata=metadata,
            )
        }
        self._provides = {Key(f"targets.{self._dest_key}", ValueTarget[Scalar]): "new"}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures logits exist, are tensors, and match representation specs."""
        from core.validation import assert_in_blackboard

        assert_in_blackboard(blackboard, f"predictions.{self._logits_key}")

        logits = blackboard.predictions[self._logits_key]
        self._representation.validate_logits(logits)

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """Compute expected value from logits and write to targets."""
        # execute() trusts validate() in debug mode, but we still use safe access
        logits: torch.Tensor = blackboard.predictions[self._logits_key]

        # Shape: [B, T, bins] or [B, bins] — to_expected_value handles both.
        scalar: torch.Tensor = self._representation.to_expected_value(logits)
        # Shape: [B, T] or [B]
        self._representation.validate_expected_value(scalar)

        return {f"targets.{self._dest_key}": scalar}


# ---------------------------------------------------------------------------
# OneHotPolicyTargetComponent
# ---------------------------------------------------------------------------


class OneHotPolicyTargetComponent(PipelineComponent):
    """Generator: Converts action indices into one-hot policy distributions.

    This ensures policy losses (e.g. for Behavioral Cloning) receive a
    consistent [B, T, K] distribution, removing the need for magic shape-matching
    logic inside the loss components.

    Args:
        num_actions: Total number of discrete actions.
        source_key: Blackboard path to action indices (default: "data.actions").
        dest_key: Blackboard key for the one-hot target (default: "policies").
    """

    def __init__(
        self,
        num_actions: int,
        source_key: str = "data.actions",
        dest_key: str = "policies",
    ) -> None:
        self._representation = ClassificationRepresentation(num_actions)
        self._source_key = source_key
        self._dest_key = dest_key

        # Deterministic contracts
        self._requires = {Key(self._source_key, Action)}
        self._provides = {Key(f"targets.{self._dest_key}", Policy[Probs]): "new"}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """Ensures action indices exist and are correctly shaped."""
        from core.validation import (
            assert_in_blackboard,
            assert_is_tensor,
            assert_shape_sanity,
        )

        assert_in_blackboard(blackboard, self._source_key)

        indices = resolve_blackboard_path(blackboard, self._source_key)
        assert_is_tensor(indices, msg=f"for {self.__class__.__name__}")
        # Indices are usually [B], [B, T], or [B, T, 1]
        assert_shape_sanity(
            indices, min_ndim=1, max_ndim=3, msg=f"for {self.__class__.__name__}"
        )

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        """Read indices from source, convert to one-hot, and write to dest."""
        indices = resolve_blackboard_path(blackboard, self._source_key)

        # Standardisation: [B, T, 1] -> [B, T]
        if indices.ndim == 3 and indices.shape[-1] == 1:
            indices = indices.squeeze(-1)

        # Use format_target for built-in validation
        one_hot = self._representation.format_target(
            {self._dest_key: indices}, target_key=self._dest_key
        )
        return {f"targets.{self._dest_key}": one_hot}
