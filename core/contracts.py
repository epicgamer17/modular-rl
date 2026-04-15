from enum import Enum
from typing import Type, Dict, Any, Optional, Tuple, List, Union
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


@dataclass(frozen=True)
class Categorical(Structure):
    bins: int

    def __repr__(self) -> str:
        return f"Categorical(bins={self.bins})"


@dataclass(frozen=True)
class Quantile(Structure):
    # TODO: what is this actually used for?
    n: int

    def __repr__(self) -> str:
        return f"Quantile(n={self.n})"


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

    def __class_getitem__(
        cls, structure: Union[Type[Structure], Structure]
    ) -> Type["SemanticType"]:
        if isinstance(structure, type):
            if issubclass(structure, (Scalar, Logits, Probs, LogProbs)):
                structure = structure()
            else:
                # For Categorical/Quantile, we expect an instance or we can't know bins/n
                raise TypeError(
                    f"Structure type {structure} must be instantiated (e.g. {structure.__name__}(...))"
                )

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
        has_time:      Whether the tensor carries a time/sequence dimension.
                       None = unspecified, True = has time, False = no time.
        time_dim:      Axis index of the time dimension (typically 1 for [B, T, *]).
        feature_shape: Shape of the non-batch, non-time dimensions
                       (e.g. (9,) for a 9-action policy vector).
        symbolic:      Symbolic dimension names for documentation/validation.
                       e.g., ("B", "T", "C") means [Batch, Time, Channels].
        dtype:         Expected torch dtype as string (e.g., "float32", "int64").
                       Note: stored as string since torch.dtype is not hashable.
    """

    ndim: Optional[int] = None
    has_time: Optional[bool] = None
    time_dim: Optional[int] = None
    feature_shape: Optional[Tuple[int, ...]] = None
    symbolic: Optional[Tuple[str, ...]] = None
    dtype: Optional[str] = None


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
                f"ndim mismatch: consumer requires rank {c.ndim}, provider declares {p.ndim}"
            )

    # Stage 4.2: Time dimension presence and position
    if c.has_time is not None:
        if p.has_time is None:
            issues.append(
                f"time dimension gap: consumer requires has_time={c.has_time}, but provider does not declare it"
            )
        elif c.has_time != p.has_time:
            issues.append(
                f"time dimension mismatch: consumer requires has_time={c.has_time}, provider declares {p.has_time}"
            )

    if c.time_dim is not None:
        if p.time_dim is None:
            issues.append(
                f"time_dim gap: consumer requires dim {c.time_dim}, but provider does not declare it"
            )
        elif c.time_dim != p.time_dim:
            issues.append(
                f"time_dim position mismatch: consumer expects dim {c.time_dim}, "
                f"provider declares dim {p.time_dim}"
            )

    # Stage 4.3: Feature shape and Safe Broadcasting
    if c.feature_shape is not None and p.feature_shape is not None:
        if c.feature_shape != p.feature_shape:
            # Check for broadcasting compatibility:
            # Rule: Dimensions must match or one of them must be 1.
            # We assume tensors are right-aligned (standard PyTorch broadcasting).
            p_feat = p.feature_shape
            c_feat = c.feature_shape

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
                    f"shape mismatch (unsafe broadcasting): consumer expects {c.feature_shape}, "
                    f"but provider provides {p.feature_shape} which is not broadcast-compatible."
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

    Identity (hash/equality) is determined solely by ``path`` and ``semantic_type``.
    The ``metadata`` and ``shape`` fields are contract annotations used by
    the DAG validator but do not affect key identity.
    """

    path: str
    semantic_type: Type[SemanticType]
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    shape: Optional[ShapeContract] = field(default=None, compare=False)

    def __hash__(self) -> int:
        """Hash the key based on its stable identity fields (path and semantic_type)."""
        return hash((self.path, self.semantic_type))

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
            if self.shape.has_time is not None:
                shape_parts.append(f"has_time={self.shape.has_time}")
            if self.shape.feature_shape is not None:
                shape_parts.append(f"features={self.shape.feature_shape}")
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
