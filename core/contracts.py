from typing import Protocol, runtime_checkable, Type, Dict, Any, Optional
from dataclasses import dataclass, field


@runtime_checkable
class SemanticType(Protocol):
    """
    Base for all semantic types in the RL pipeline.
    These types define meaning, not structure (e.g., shapes).
    """

    pass


@dataclass(frozen=True)
class Key:
    """A semantic key that combines a blackboard path, a semantic type, and optional metadata."""

    path: str
    semantic_type: Type[SemanticType]
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __hash__(self) -> int:
        """Hash the key based on its stable fields (path and semantic_type)."""
        return hash((self.path, self.semantic_type))

    def __str__(self) -> str:
        meta_str = f", metadata={self.metadata}" if self.metadata else ""
        return f"Key({self.path}, {self.semantic_type.__name__}{meta_str})"


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
