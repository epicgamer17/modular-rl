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
    Partial shape schema for DAG-time validation of tensor structure based on semantic axis names.

    All fields are optional — unspecified fields impose no constraint.
    When both a provider and consumer specify a field, the DAG validator
    checks compatibility at graph-build time.

    Semantic Axis Names:
        "B": Batch dimension (strict).
        "T": Time/Sequence dimension (strict).
        "F": Feature dimension (strict).
        "A": Action dimension (strict).
        "C": Channel dimension (strict).
        "H": Height dimension (strict).
        "W": Width dimension (strict).
        "*": Broadcastable / Optional / Wildcard dimension.
        (Any other uppercase letter is allowed for custom descriptive labeling).

    Fields:
        semantic_shape: Tuple of semantic axis names (e.g., ("B", "T", "F")).
        time_val:       Explicit expected size of the "T" dimension.
        event_shape:    Explicit expected size of all payload dimensions.
                        Must match the number of non-B/T axes in semantic_shape.
        dtype:          Expected torch dtype (e.g., torch.float32, torch.int64).
    """

    semantic_shape: Optional[Tuple[str, ...]] = None
    time_val: Optional[int] = None
    event_shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[torch.dtype] = None

    def __post_init__(self) -> None:
        """Enforce internal consistency of the shape contract."""
        if self.semantic_shape is not None:
            # Ensure "B" (Batch) is always present at index 0
            assert self.semantic_shape[0] == "B", f"Semantic shape must start with 'B' (Batch), got {self.semantic_shape}"
            
            # Count descriptive axes (anything that isn't B, T, or *)
            # These map 1-to-1 to the event_shape tuple.
            num_event_axes = sum(1 for axis in self.semantic_shape if axis not in ("B", "T", "*"))

            # Check consistency between semantic_shape and event_shape
            if self.event_shape is not None:
                assert len(self.event_shape) == num_event_axes, (
                    f"ShapeContract inconsistency: semantic_shape {self.semantic_shape} has {num_event_axes} descriptive axes, "
                    f"but event_shape {self.event_shape} has length {len(self.event_shape)}."
                )
            
            # Check consistency between semantic_shape and time_val
            if self.time_val is not None:
                has_time_axis = any(axis == "T" for axis in self.semantic_shape)
                assert has_time_axis, (
                    f"ShapeContract inconsistency: time_val {self.time_val} specified, "
                    f"but semantic_shape {self.semantic_shape} has no 'T' axis."
                )

    def format_shape(self) -> str:
        """Return a human-readable shape string using semantic names and explicit sizes."""
        if self.semantic_shape is None:
            return "opaque"

        parts = []
        event_idx = 0
        for axis in self.semantic_shape:
            if axis == "B":
                parts.append("B")
            elif axis == "T":
                if self.time_val is not None:
                    parts.append(f"T={self.time_val}")
                else:
                    parts.append("T")
            elif axis == "*":
                parts.append("*")
            else:
                # Any other axis (A, F, C, H, W, etc.) counts as an 'event' dimension
                if self.event_shape is not None and event_idx < len(self.event_shape):
                    # Use the explicit value from event_shape
                    parts.append(str(self.event_shape[event_idx]))
                    event_idx += 1
                else:
                    parts.append(axis)

        return f"({', '.join(parts)})"


def check_shape_compatibility(provider: "Key", consumer: "Key") -> List[str]:
    """
    Check whether a provider Key's shape contract satisfies a consumer's requirements.

    Returns a list of human-readable incompatibility strings (empty = compatible).
    Checks are opt-in: if the consumer field is None, that field is skipped.
    """
    c = consumer.shape
    if c is None:
        return []

    p = provider.shape
    if p is None:
        return [
            f"contract gap: consumer requires {c.format_shape()}, but provider is opaque (no shape contract)"
        ]

    issues: List[str] = []

    # Stage 1: Semantic Shape Integrity (Gap + Rank check)
    if c.semantic_shape is not None:
        if p.semantic_shape is None:
            issues.append(
                f"semantic_shape gap: consumer requires {c.semantic_shape}, but provider does not declare one"
            )
            return issues
        elif len(c.semantic_shape) != len(p.semantic_shape):
            issues.append(
                f"Rank mismatch: consumer requires rank {len(c.semantic_shape)} {c.semantic_shape}, "
                f"but provider gives rank {len(p.semantic_shape)} {p.semantic_shape}"
            )
            return issues # Cannot perform axis-by-axis check if ranks differ

    # Stage 1 & 3: Axis-by-Axis Label and Size consistency
    if c.semantic_shape is not None and p.semantic_shape is not None:
        c_event_idx = 0
        p_event_idx = 0
        
        for i, (c_axis, p_axis) in enumerate(zip(c.semantic_shape, p.semantic_shape)):
            # 1. Resolve labels and sizes for this axis
            c_val = None
            if c_axis == "T":
                c_val = c.time_val
            elif c_axis not in ("B", "T", "*"):
                if c.event_shape and c_event_idx < len(c.event_shape):
                    c_val = c.event_shape[c_event_idx]
            
            p_val = None
            if p_axis == "T":
                p_val = p.time_val
            elif p_axis not in ("B", "T", "*"):
                if p.event_shape and p_event_idx < len(p.event_shape):
                    p_val = p.event_shape[p_event_idx]
            
            # 2. Advance event indices independently
            if c_axis not in ("B", "T", "*"):
                c_event_idx += 1
            if p_axis not in ("B", "T", "*"):
                p_event_idx += 1

            # 3. Check compatibility
            # BROADCASTING RULE:
            # If consumer uses '*', it explicitly allows any size (it's a wildcard).
            # If consumer uses a named axis (B, T, F, A, etc.), sizes must match EXACTLY.
            if c_axis == "*":
                continue 

            # Gap check: if provider has no event_shape, we preserve original heuristic for 'A'
            if c_val is not None and p.event_shape is None and p_axis == "A":
                issues.append(
                    f"event_shape gap at dim {i} ('{c_axis}'): consumer requires size {c_val}, but provider does not declare an event_shape."
                )

            # Check label mismatch (if provider isn't also a wildcard)
            if p_axis != "*" and c_axis != p_axis:
                issues.append(
                    f"Semantic axis mismatch at dim {i}: consumer expects '{c_axis}', provider gives '{p_axis}'"
                )
            
            # Check size mismatch (if both specify a size)
            if c_val is not None and p_val is not None:
                if c_val != p_val:
                    issues.append(
                        f"Dimension mismatch at dim {i} ('{c_axis}'): "
                        f"consumer requires size {c_val}, provider gives {p_val}. "
                        f"(Broadcasting only allowed if consumer uses '*' for this axis)"
                    )

    # Stage 2: Catch-all for time value if T wasn't matched in loop (shouldn't happen with same rank)
    # But if ranks differ, we might still want to check.
    # Actually, current code handles it inside the loop better.

    # Stage 4: Dtype check
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
