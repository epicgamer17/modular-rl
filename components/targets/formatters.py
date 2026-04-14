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
from typing import TYPE_CHECKING, Optional, Type, Set
from core.contracts import Key, ValueTarget, Reward, ActionDistribution, PolicyLogits, SemanticType

from core import PipelineComponent, Blackboard
from core.path_resolver import resolve_blackboard_path
from core.contracts import Key, ValueTarget, PolicyLogits, ActionDistribution, Action, Reward
from modules.representations import BaseRepresentation, DiscreteSupportRepresentation, ClassificationRepresentation

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
        self._semantic_type = semantic_type
        
        # Deterministic contracts
        self._requires = {Key(self._source_key, self._semantic_type)}
        self._provides = {Key(f"targets.{self._dest_key}", self._semantic_type)}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        pass

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
        semantic_type: Type[SemanticType] = SemanticType,
    ) -> None:
        self._source_key = source_key
        self._dest_key = dest_key
        self._representation = representation
        self._semantic_type = semantic_type
        
        # Deterministic contracts
        self._requires = {Key(self._source_key, self._semantic_type)}
        self._provides = {Key(f"targets.{self._dest_key}", self._semantic_type)}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        pass

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
        semantic_type: Type[SemanticType] = SemanticType,
    ) -> None:
        self._source_key = source_key
        self._dest_key = dest_key
        self._representation = representation
        self._semantic_type = semantic_type
        
        # Deterministic contracts
        self._requires = {Key(self._source_key, self._semantic_type)}
        self._provides = {Key(f"targets.{self._dest_key}", self._semantic_type)}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        pass

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
        
        # Deterministic contracts
        self._requires = {Key(f"predictions.{self._logits_key}", PolicyLogits)}
        self._provides = {Key(f"targets.{self._dest_key}", ValueTarget)}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        assert self._logits_key in blackboard.predictions

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
        self._provides = {Key(f"targets.{self._dest_key}", ActionDistribution)}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Set[Key]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> None:
        """Read indices from source, convert to one-hot, and write to dest."""
        indices = resolve_blackboard_path(blackboard, self._source_key)
        
        # Standardisation: [B, T, 1] -> [B, T]
        if indices.ndim == 3 and indices.shape[-1] == 1:
            indices = indices.squeeze(-1)
            
        # [B, T] -> [B, T, K]
        one_hot = self._representation.to_representation(indices)
        blackboard.targets[self._dest_key] = one_hot
