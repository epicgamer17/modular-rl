"""
Type specifications and operator contracts for the RL IR.
Defines types for ports and verifies compatibility between nodes.
"""

from dataclasses import dataclass
from typing import Dict, Union, Any, Tuple

from core.schema import TensorSpec, Schema, Field

# Type aliases for spec objects
Spec = Union[TensorSpec, Schema]


@dataclass(frozen=True)
class OperatorSpec:
    """
    Defines the data contract for an operator.

    Attributes:
        inputs: Mapping of input port names to their required Specs.
        outputs: Mapping of output port names to their produced Specs.
    """

    inputs: Dict[str, Spec]
    outputs: Dict[str, Spec]

    @classmethod
    def create(
        cls, inputs: Dict[str, Spec], outputs: Union[Spec, Dict[str, Spec]]
    ) -> "OperatorSpec":
        """Helper to create a spec, allowing single output to be passed directly."""
        if not isinstance(outputs, dict):
            outputs = {"default": outputs}
        return cls(inputs=inputs, outputs=outputs)


# Helper functions to create specs with a cleaner syntax
def Tensor(shape: Tuple[int, ...], dtype: str) -> TensorSpec:
    """Creates a TensorSpec with specified shape and dtype."""
    return TensorSpec(shape=shape, dtype=dtype)


def Scalar(dtype: str) -> TensorSpec:
    """Creates a 0-d TensorSpec (scalar) with specified dtype."""
    return TensorSpec(shape=(), dtype=dtype)


# Common RL-specific types
# Rank-aware types to distinguish between single step and batched execution
SingleObs = TensorSpec(shape=(-1,), dtype="float32", tags=["obs", "single"])  # [D]
BatchObs = TensorSpec(shape=(-1, -1), dtype="float32", tags=["obs", "batch"])  # [B, D]

SingleQ = TensorSpec(shape=(-1,), dtype="float32", tags=["q_values", "single"])  # [A]
BatchQ = TensorSpec(
    shape=(-1, -1), dtype="float32", tags=["q_values", "batch"]
)  # [B, A]

# TODO: Remove these.
# Alias legacy names for backward compatibility if needed,
# but encourage use of rank-explicit names.
ObsTensor = SingleObs
ActionValuesTensor = SingleQ
ScalarLoss = Scalar("float32")

TransitionBatch = Schema(
    fields=[
        Field("obs", BatchObs),
        Field("action", TensorSpec(shape=(-1,), dtype="int64", tags=["action"])),
        Field("reward", TensorSpec(shape=(-1,), dtype="float32", tags=["reward"])),
        Field("next_obs", BatchObs),
        Field("done", TensorSpec(shape=(-1,), dtype="bool", tags=["done"])),
    ]
)

# Registry of specifications for built-in and custom operators
_SPEC_REGISTRY: Dict[str, OperatorSpec] = {}


def register_spec(node_type: str, spec: OperatorSpec) -> None:
    """Registers a specification for a given node type."""
    _SPEC_REGISTRY[node_type] = spec


def get_spec(node_type: str) -> str | OperatorSpec:
    """Retrieves the specification for a node type, or None if not registered."""
    return _SPEC_REGISTRY.get(node_type)


def is_compatible(src: Spec, dst: Spec) -> bool:
    """
    Checks if a source spec is compatible with a destination spec.

    Rules:
    - Both must be TensorSpecs or both must be Schemas.
    - If TensorSpecs: shapes and dtypes must match.
    - If Schemas: field sets, shapes, and dtypes must match.
    """
    if isinstance(src, TensorSpec) and isinstance(dst, TensorSpec):
        # We allow -1 to match any dimension for now (simple broadcast/flexible batch check)
        if len(src.shape) != len(dst.shape):
            return False
        for s, d in zip(src.shape, dst.shape):
            if s != -1 and d != -1 and s != d:
                return False
        return src.dtype == dst.dtype

    if isinstance(src, Schema) and isinstance(dst, Schema):
        return src.is_compatible(dst)

    return False


def format_spec(spec: Spec) -> str:
    """Returns a human-readable string representation of a Spec."""
    # Prioritize semantic names for well-known RL types
    if spec == SingleObs:
        return "SingleObs"
    if spec == BatchObs:
        return "BatchObs"
    if spec == SingleQ:
        return "SingleQ"
    if spec == BatchQ:
        return "BatchQ"
    if spec == ScalarLoss:
        return "ScalarLoss"

    if isinstance(spec, TensorSpec):
        tag_info = f" ({', '.join(spec.tags)})" if spec.tags else ""
        return f"Tensor{list(spec.shape)}[{spec.dtype}]{tag_info}"

    if isinstance(spec, Schema):
        # Determine name (check if it matches a known constant)
        name = "Schema"
        if spec == TransitionBatch:
            name = "TransitionBatch"

        field_strs = []
        for f in spec.fields:
            field_strs.append(f"  {f.name}: {list(f.spec.shape)} {f.spec.dtype}")
        fields_summary = "\n".join(field_strs)
        return f"{name}[\n{fields_summary}\n]"

    return str(spec)
