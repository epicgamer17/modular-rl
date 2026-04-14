from typing import Protocol, runtime_checkable, Type
from dataclasses import dataclass

@runtime_checkable
class SemanticType(Protocol):
    """
    Base for all semantic types in the RL pipeline.
    These types define meaning, not structure (e.g., shapes).
    """
    pass

@dataclass(frozen=True)
class Key:
    """A semantic key that combines a blackboard path with a semantic type."""
    path: str
    semantic_type: Type[SemanticType]

    def __str__(self):
        return f"Key({self.path}, {self.semantic_type.__name__})"


class Observation(SemanticType): pass
class Action(SemanticType): pass
class PolicyLogits(SemanticType): pass
class ActionDistribution(SemanticType): pass
class ValueEstimate(SemanticType): pass
class ValueTarget(SemanticType): pass
class Reward(SemanticType): pass
class Trajectory(SemanticType): pass
class HiddenState(SemanticType): pass
class LossScalar(SemanticType): pass
class ToPlay(SemanticType): pass
class Done(SemanticType): pass
class Advantage(SemanticType): pass
class Return(SemanticType): pass
class Mask(SemanticType): pass
