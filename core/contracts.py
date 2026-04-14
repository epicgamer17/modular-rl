from typing import Protocol, runtime_checkable, Type, Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field


@runtime_checkable
class SemanticType(Protocol):
    """
    Base for all semantic types in the RL pipeline.
    These types define meaning, not structure (e.g., shapes).
    """

    pass


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
        distribution:  What the feature dimension represents:
                       "logits"  — raw unnormalised log-probabilities
                       "probs"   — normalised probabilities (sum-to-1)
                       "log_probs" — log-probabilities
                       "scalar"  — single float value per sample
                       "atoms_N" — distributional with N support atoms (e.g. "atoms_51")
    """

    ndim: Optional[int] = None
    has_time: Optional[bool] = None
    time_dim: Optional[int] = None
    feature_shape: Optional[Tuple[int, ...]] = None
    distribution: Optional[str] = None


def check_shape_compatibility(provider: "Key", consumer: "Key") -> List[str]:
    """
    Check whether a provider Key's shape contract satisfies a consumer's requirements.

    Returns a list of human-readable incompatibility strings (empty = compatible).
    Checks are opt-in: if either side leaves a field as None, that field is skipped.
    """
    p = provider.shape
    c = consumer.shape

    # No constraints declared on either side — nothing to check.
    if c is None or p is None:
        return []

    issues: List[str] = []

    # ndim
    if c.ndim is not None and p.ndim is not None and c.ndim != p.ndim:
        issues.append(
            f"ndim mismatch: consumer expects {c.ndim}, provider declares {p.ndim}"
        )

    # Time dimension presence
    if c.has_time is True and p.has_time is False:
        issues.append(
            "time dimension mismatch: consumer expects a time dimension, "
            "provider explicitly declares none"
        )
    if c.has_time is False and p.has_time is True:
        issues.append(
            "time dimension mismatch: consumer expects no time dimension, "
            "provider declares one"
        )

    # Time dimension position
    if c.time_dim is not None and p.time_dim is not None and c.time_dim != p.time_dim:
        issues.append(
            f"time_dim position mismatch: consumer expects dim {c.time_dim}, "
            f"provider declares dim {p.time_dim}"
        )

    # Feature shape
    if c.feature_shape is not None and p.feature_shape is not None:
        if c.feature_shape != p.feature_shape:
            issues.append(
                f"feature_shape mismatch: consumer expects {c.feature_shape}, "
                f"provider declares {p.feature_shape}"
            )

    # Distribution kind
    if c.distribution is not None and p.distribution is not None:
        if c.distribution != p.distribution:
            issues.append(
                f"distribution mismatch: consumer expects '{c.distribution}', "
                f"provider declares '{p.distribution}'"
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
            parts.append(f"shape={self.shape}")
        return f"Key({', '.join(parts)})"


class Observation(SemanticType):
    pass


class Action(SemanticType):
    pass


class PolicyLogits(SemanticType):
    pass


class ActionDistribution(SemanticType):
    pass


class ValueEstimate(SemanticType):
    pass


class ValueTarget(SemanticType):
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
