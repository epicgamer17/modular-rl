from enum import Enum
from typing import Type, Dict, Any, Optional, Tuple, List, Union
import torch
from dataclasses import dataclass, field


class WriteMode(Enum):
    """Write mode for pipeline component provides."""

    NEW = "new"
    OVERWRITE = "overwrite"
    APPEND = "append"
    OPTIONAL = "optional"


@dataclass(frozen=True)
class Structure:
    """Base for semantic structures."""

    pass


@dataclass(frozen=True)
class Scalar(Structure):
    def __repr__(self) -> str:
        return "Scalar"


@dataclass(frozen=True)
class Logits(Structure):
    def __repr__(self) -> str:
        return "Logits"


@dataclass(frozen=True)
class Probs(Structure):
    def __repr__(self) -> str:
        return "Probs"


@dataclass(frozen=True)
class LogProbs(Structure):
    def __repr__(self) -> str:
        return "LogProbs"


_STRUCTURED_TYPE_CACHE: Dict[
    Tuple[Type["SemanticType"], Structure], Type["SemanticType"]
] = {}


class SemanticType:
    """
    Base for all semantic types in the RL pipeline.
    These types define meaning, not structure (e.g., shapes),
    but can be parameterized with a Structure (e.g. ValueEstimate[Scalar]).
    """

    structure: Optional[Structure] = None
    
    @classmethod
    def is_compatible(cls, other: Type["SemanticType"]) -> bool:
        """
        Check if this semantic type (the requirement) is compatible with another (the provider).
        Logic:
        1. Both types must share a common base type (inheritance check).
        2. If the requirement (cls) is abstract (structure is None), it accepts any representation.
        3. If the requirement (cls) is concrete, the provider must match that structure exactly.
        """
        # Same base semantic type (one must be a subclass of the other)
        if not issubclass(other, cls) and not issubclass(cls, other):
            return False

        s_req = getattr(cls, "structure", None)
        s_prov = getattr(other, "structure", None)

        # If requirement is abstract -> allow any representation of this semantic type
        if s_req is None:
            return True

        # If requirement is concrete -> provider must match exactly
        return s_req == s_prov

    def __class_getitem__(
        cls, structure: Union[Type[Structure], Structure]
    ) -> Type["SemanticType"]:
        if isinstance(structure, type):
            # All current structures (Scalar, Logits, Probs, LogProbs) are stateless
            structure = structure()

        cache_key = (cls, structure)
        if cache_key not in _STRUCTURED_TYPE_CACHE:
            _STRUCTURED_TYPE_CACHE[cache_key] = type(
                f"{cls.__name__}[{structure}]", (cls,), {"structure": structure}
            )
        return _STRUCTURED_TYPE_CACHE[cache_key]


@dataclass(frozen=True)
class ShapeContract:
    """
    Partial shape schema for DAG-time validation of tensor structure.

    All fields are optional — unspecified fields impose no constraint.
    When both a provider and consumer specify a field, the DAG validator
    checks compatibility at graph-build time, catching "types match but
    behavior is wrong" bugs before the first training step.

    Fields:
        ndim:          Expected tensor rank (e.g. 2 for [B, A], 3 for [B, T, A]).
        time_dim:      Axis index of the time dimension (typically 1 for [B, T, *]).
                       None = no time/sequence dimension.
        event_shape:   Shape of the non-batch, non-time dimensions
                       (e.g. (9,) for a 9-action policy vector).
                       Excludes batch and time dimensions.
        symbolic:      Symbolic dimension names for documentation/validation.
                       e.g., ("B", "T", "C") means [Batch, Time, Channels].
        dtype:         Expected torch dtype (e.g., torch.float32, torch.int64).
    """

    ndim: Optional[int] = None
    time_dim: Optional[int] = None
    event_shape: Optional[Tuple[int, ...]] = None
    symbolic: Optional[Tuple[str, ...]] = None
    dtype: Optional[torch.dtype] = None

    def __post_init__(self) -> None:
        """Enforce internal consistency of the shape contract."""
        if self.ndim is not None:
            assert self.ndim > 0, f"ndim must be positive, got {self.ndim}"

        if self.time_dim is not None:
            if self.ndim is not None:
                assert self.time_dim < self.ndim, (
                    f"time_dim {self.time_dim} out of bounds for ndim {self.ndim}"
                )
            assert self.time_dim != 0, "time_dim cannot be 0 (reserved for batch dimension)"

        if self.event_shape is not None:
            if self.ndim is not None:
                expected_ndim = len(self.event_shape) + 1
                if self.time_dim is not None:
                    expected_ndim += 1
                assert self.ndim == expected_ndim, (
                    f"ShapeContract inconsistency: ndim={self.ndim} does not match "
                    f"event_shape={self.event_shape} (len={len(self.event_shape)}) + "
                    f"batch(1) + time({1 if self.time_dim is not None else 0}). "
                    f"Expected ndim {expected_ndim}."
                )

        if self.symbolic is not None:
            if self.ndim is not None:
                assert len(self.symbolic) == self.ndim, (
                    f"Symbolic dims {self.symbolic} (len={len(self.symbolic)}) "
                    f"must match ndim {self.ndim}"
                )
            if self.event_shape is not None:
                expected_len = len(self.event_shape) + 1
                if self.time_dim is not None:
                    expected_len += 1
                assert len(self.symbolic) == expected_len, (
                    f"Symbolic dims {self.symbolic} (len={len(self.symbolic)}) "
                    f"must match event_shape {self.event_shape} + batch/time (expected {expected_len})"
                )

    def format_shape(self) -> str:
        """Return a human-readable shape string using symbolic names if available."""
        if self.symbolic:
            return f"({', '.join(self.symbolic)})"

        # Fallback to building from ndim/event_shape/time_dim
        parts = ["B"]
        if self.time_dim is not None:
            parts.append("T")

        if self.event_shape is not None:
            for s in self.event_shape:
                parts.append(str(s))
        elif self.ndim is not None:
            # Add placeholders for unknown dims
            needed = self.ndim - len(parts)
            for _ in range(needed):
                parts.append("?")

        return f"({', '.join(parts)})"


def check_shape_compatibility(provider: "Key", consumer: "Key") -> List[str]:
    """
    Check whether a provider Key's shape contract satisfies a consumer's requirements.

    Returns a list of human-readable incompatibility strings (empty = compatible).
    Checks are opt-in: if the consumer field is None, that field is skipped.
    If the consumer HAS a requirement but the provider is None, it is reported as a gap.
    """
    c = consumer.shape
    if c is None:
        return []

    p = provider.shape
    if p is None:
        return [
            f"contract gap: consumer requires {c}, but provider is opaque (no shape contract)"
        ]

    issues: List[str] = []

    # Stage 4.1: Rank and required dimensions existence
    if c.ndim is not None:
        if p.ndim is None:
            issues.append(
                f"ndim gap: consumer requires rank {c.ndim}, but provider does not declare one"
            )
        elif c.ndim != p.ndim:
            issues.append(
                f"Rank mismatch: consumer requires {c.ndim} dimensions, but provider gives {p.ndim}"
            )

    # Stage 4.2: Time dimension presence and position
    if c.time_dim is not None:
        if p.time_dim is None:
            issues.append(
                f"Time dimension gap: consumer requires sequence dim at {c.time_dim}, but provider declares no sequence dimension"
            )
        elif c.time_dim != p.time_dim:
            issues.append(
                f"Time dimension position mismatch: consumer expects sequence dim at {c.time_dim}, "
                f"provider declares sequence dim at {p.time_dim}"
            )
    elif p.time_dim is not None:
        # Consumer explicitly expects NO time dimension (time_dim=None), but provider HAS one
        issues.append(
            f"Time dimension mismatch: provider has a sequence dimension (T), but consumer does not expect one"
        )

    # Stage 4.3: Event shape and Safe Broadcasting
    if c.event_shape is not None and p.event_shape is not None:
        if c.event_shape != p.event_shape:
            # Check for broadcasting compatibility:
            # Rule: Dimensions must match or one of them must be 1.
            # We assume tensors are right-aligned (standard PyTorch broadcasting).
            p_feat = p.event_shape
            c_feat = c.event_shape

            is_compatible = True
            if len(p_feat) > len(c_feat):
                is_compatible = False
            else:
                # Pad p_feat with 1s on the left to match length
                p_padded = (1,) * (len(c_feat) - len(p_feat)) + p_feat
                for p_dim, c_dim in zip(p_padded, c_feat):
                    if p_dim != c_dim and p_dim != 1 and c_dim != 1:
                        is_compatible = False
                        break

            if not is_compatible:
                issues.append(
                    f"shape mismatch (unsafe broadcasting): consumer expects {c.event_shape}, "
                    f"but provider provides {p.event_shape} which is not broadcast-compatible."
                )

    # Stage 4.4: Symbolic Dimensions Consistency
    if c.symbolic is not None:
        if p.symbolic is None:
            issues.append(
                f"symbolic gap: consumer requires {c.symbolic}, but provider does not declare symbolic names"
            )
        elif len(c.symbolic) != len(p.symbolic):
            issues.append(
                f"symbolic rank mismatch: consumer has {len(c.symbolic)} dims {c.symbolic}, "
                f"provider has {len(p.symbolic)} dims {p.symbolic}"
            )
        elif c.symbolic != p.symbolic:
            issues.append(
                f"symbolic mismatch: consumer expects {c.symbolic}, "
                f"provider declares {p.symbolic}"
            )

    # Stage 4.5: Dtype check
    if c.dtype is not None:
        if p.dtype is None:
            issues.append(
                f"dtype gap: consumer requires {c.dtype}, but provider does not declare one"
            )
        elif c.dtype != p.dtype:
            issues.append(
                f"dtype mismatch: consumer expects {c.dtype}, provider declares {p.dtype}"
            )

    return issues


@dataclass(frozen=True)
class Key:
    """
    A semantic contract that binds a blackboard path to a semantic type,
    optional metadata, and an optional shape contract.

    Identity (hash/equality) is determined solely by ``path``.
    The ``semantic_type``, ``metadata`` and ``shape`` fields are contract annotations 
    used by the DAG validator but do not affect key identity.
    """

    path: str
    semantic_type: Type[SemanticType] = field(compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    shape: Optional[ShapeContract] = field(default=None, compare=False)

    def __hash__(self) -> int:
        """Hash the key based on its stable identity field (path)."""
        return hash(self.path)

    def __str__(self) -> str:
        parts = [self.path, self.semantic_type.__name__]
        if self.metadata:
            parts.append(f"metadata={self.metadata}")
        if self.shape is not None:
            shape_parts = []
            if self.shape.symbolic:
                shape_parts.append(f"symbolic={self.shape.symbolic}")
            if self.shape.dtype:
                shape_parts.append(f"dtype={self.shape.dtype}")
            if self.shape.ndim is not None:
                shape_parts.append(f"ndim={self.shape.ndim}")
            if self.shape.time_dim is not None:
                shape_parts.append(f"time_dim={self.shape.time_dim}")
            if self.shape.event_shape is not None:
                shape_parts.append(f"event={self.shape.event_shape}")
            parts.append(f"shape({', '.join(shape_parts)})")
        return f"Key({', '.join(parts)})"


class Observation(SemanticType):
    pass


class Action(SemanticType):
    pass


class Policy(SemanticType):
    pass


class ValueEstimate(SemanticType):
    pass


class ValueTarget(SemanticType):
    pass


class QValues(SemanticType):
    pass


class QTargets(SemanticType):
    pass


class Reward(SemanticType):
    pass


class Trajectory(SemanticType):
    pass


class HiddenState(SemanticType):
    pass


class LossScalar(SemanticType):
    pass


class ToPlay(SemanticType):
    pass


class Done(SemanticType):
    pass


class Advantage(SemanticType):
    pass


class LogProb(SemanticType):
    pass


class Return(SemanticType):
    pass


class Mask(SemanticType):
    pass


class Priority(SemanticType):
    pass


class GradientScale(SemanticType):
    pass


class Weight(SemanticType):
    pass


class Epsilon(SemanticType):
    pass


class Metric(SemanticType):
    pass
