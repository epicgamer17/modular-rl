"""
Type system for the RL IR.
Defines semantic types for tensors and other data structures in RL.
"""

from dataclasses import dataclass, field
import copy
from typing import Dict, List, Any, Tuple, Optional, Set, Union
from enum import Enum, auto

class RLTypeCategory(Enum):
    TENSOR = auto()
    TRAJECTORY = auto()
    EPISODE = auto()
    DISTRIBUTION = auto()
    POLICY_SNAPSHOT = auto()
    REPLAY_BATCH = auto()
    SCALAR_METRIC = auto()
    RNG_KEY = auto()
    HIDDEN_STATE = auto()

@dataclass(frozen=True, kw_only=True)
class RLType:
    """Base class for all RL types."""
    category: RLTypeCategory
    tags: Set[str] = field(default_factory=set)

    def is_compatible(self, other: 'RLType') -> bool:
        """Check if this type is compatible with another."""
        # Check if basic category matches
        if self.category != other.category:
            return False
        
        # Check semantic tags compatibility
        # If one is OnPolicy and other is OffPolicy, they are incompatible
        from core.schema import TAG_ON_POLICY, TAG_OFF_POLICY
        if TAG_ON_POLICY in self.tags and TAG_OFF_POLICY in other.tags:
            return False
        if TAG_OFF_POLICY in self.tags and TAG_ON_POLICY in other.tags:
            return False
            
        return True

    def vectorize(self) -> 'RLType':
        """Returns a batched version of this type."""
        return copy.deepcopy(self)

@dataclass(frozen=True, kw_only=True)
class TensorType(RLType):
    """
    Represents a tensor with semantic dimensions.
    Example: Tensor[B, T, D]
    """
    shape: Tuple[Union[int, str], ...]
    dtype: str
    category: RLTypeCategory = RLTypeCategory.TENSOR

    def is_compatible(self, other: 'RLType') -> bool:
        if not super().is_compatible(other):
            return False
        if not isinstance(other, TensorType):
            return False
        return self.dtype == other.dtype and len(self.shape) == len(other.shape)

    def vectorize(self) -> 'TensorType':
        return TensorType(
            shape=('B',) + self.shape,
            dtype=self.dtype,
            tags=self.tags | {"batched"}
        )

@dataclass(frozen=True, kw_only=True)
class DistributionType(RLType):
    """
    Represents a probability distribution.
    Example: Distribution[Categorical]
    """
    dist_type: str # e.g., 'Categorical', 'Normal'
    is_logits: bool = False
    category: RLTypeCategory = RLTypeCategory.DISTRIBUTION

    def is_compatible(self, other: 'RLType') -> bool:
        if not super().is_compatible(other):
            return False
        if not isinstance(other, DistributionType):
            return False
        return self.dist_type == other.dist_type and self.is_logits == other.is_logits

    def vectorize(self) -> 'DistributionType':
        return DistributionType(
            dist_type=self.dist_type,
            is_logits=self.is_logits,
            tags=self.tags | {"batched"}
        )

@dataclass(frozen=True, kw_only=True)
class TrajectoryType(RLType):
    """
    Represents a trajectory of length T.
    """
    length: Union[int, str]
    category: RLTypeCategory = RLTypeCategory.TRAJECTORY

@dataclass(frozen=True, kw_only=True)
class EpisodeType(RLType):
    category: RLTypeCategory = RLTypeCategory.EPISODE

@dataclass(frozen=True, kw_only=True)
class PolicySnapshotType(RLType):
    version: int = 0
    category: RLTypeCategory = RLTypeCategory.POLICY_SNAPSHOT

@dataclass(frozen=True, kw_only=True)
class ReplayBatchType(RLType):
    category: RLTypeCategory = RLTypeCategory.REPLAY_BATCH

@dataclass(frozen=True, kw_only=True)
class ScalarMetricType(RLType):
    category: RLTypeCategory = RLTypeCategory.SCALAR_METRIC

@dataclass(frozen=True, kw_only=True)
class RNGKeyType(RLType):
    category: RLTypeCategory = RLTypeCategory.RNG_KEY

@dataclass(frozen=True, kw_only=True)
class HiddenStateType(RLType):
    category: RLTypeCategory = RLTypeCategory.HIDDEN_STATE
