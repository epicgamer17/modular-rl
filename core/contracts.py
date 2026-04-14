from typing import Protocol, runtime_checkable

@runtime_checkable
class SemanticType(Protocol):
    """
    Base for all semantic types in the RL pipeline.
    These types define meaning, not structure (e.g., shapes).
    """
    pass

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
