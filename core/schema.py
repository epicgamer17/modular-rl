"""
Schema system for the RL IR.
Defines data specifications for tensors, fields, schemas, and trajectories.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional, Set

# Semantic Tag Constants
TAG_ON_POLICY = "OnPolicy"
TAG_OFF_POLICY = "OffPolicy"
TAG_ORDERED = "Ordered"
TAG_UNORDERED = "Unordered"
TAG_STOCHASTIC = "Stochastic"
TAG_DETERMINISTIC = "Deterministic"

@dataclass(frozen=True)
class TensorSpec:
    """
    Specification for a tensor-like data object.
    
    Attributes:
        shape: Dimensions of the tensor.
        dtype: Data type (e.g., 'float32', 'int64').
        tags: Semantic tags for the data.
    """
    shape: Tuple[int, ...]
    dtype: str
    tags: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class Field:
    """
    A named data field within a schema.
    """
    name: str
    spec: TensorSpec

@dataclass(frozen=True)
class Schema:
    """
    A collection of named fields defining a data structure.
    """
    fields: List[Field]

    def __post_init__(self):
        # Validate that field names are unique
        names = [f.name for f in self.fields]
        assert len(names) == len(set(names)), f"Duplicate field names in schema: {names}"

    def get_field_map(self) -> Dict[str, TensorSpec]:
        """Convert fields to a dictionary for easier lookup."""
        return {f.name: f.spec for f in self.fields}

    def is_compatible(self, other: 'Schema') -> bool:
        """
        Check if another schema is compatible with this one.
        
        Compatibility rules:
        1. Must have the same set of field names.
        2. Corresponding fields must have the same shape and dtype.
        """
        self_map = self.get_field_map()
        other_map = other.get_field_map()

        if set(self_map.keys()) != set(other_map.keys()):
            return False

        for name, spec in self_map.items():
            other_spec = other_map[name]
            if spec.shape != other_spec.shape:
                return False
            if spec.dtype != other_spec.dtype:
                return False
        
        return True

@dataclass(frozen=True)
class TrajectorySpec:
    """
    Specification for a trajectory of data.
    
    A trajectory consists of sequences of data following a specific Schema.
    """
    schema: Schema
    max_length: Optional[int] = None
    tags: List[str] = field(default_factory=list)
