"""
Type specifications and operator contracts for the RL IR.
Defines types for ports and verifies compatibility between nodes.
"""

from dataclasses import dataclass
from typing import Dict, Union, Any, Tuple, List, Set, Optional, Callable

from core.schema import TensorSpec, Schema, Field
from core.types import (
    RLType, TensorType, TrajectoryType, EpisodeType, 
    DistributionType, PolicySnapshotType, ReplayBatchType,
    ScalarMetricType, RNGKeyType, HiddenStateType
)

# Type aliases for spec objects
Spec = Union[TensorSpec, Schema]


@dataclass(frozen=True)
class PortSpec:
    """
    Detailed specification for an individual input or output port.
    """

    spec: Spec
    required: bool = True
    variadic: bool = False
    default: Any = None
    description: str = ""


@dataclass(frozen=True)
class OperatorSpec:
    """
    Defines the full metadata and data contract for an operator.
    """

    name: str
    version: str = "1.0.0"

    inputs: Dict[str, PortSpec] = None
    outputs: Dict[str, PortSpec] = None

    pure: bool = False
    stateful: bool = False
    deterministic: bool = False

    side_effects: List[str] = None

    requires_models: List[str] = None
    requires_buffers: List[str] = None
    requires_optimizer: bool = False

    allowed_contexts: Set[str] = None  # {"actor", "learner"}

    tags: Set[str] = None
    shape_fn: Optional[Callable[[Dict[str, Spec]], Dict[str, Spec]]] = None

    # Cost Model metrics
    estimated_flops: int = 0
    memory_reads: int = 0
    kernel_launch_cost: float = 0.0

    def __post_init__(self) -> None:
        """Initialize optional collections."""
        pass

    @classmethod
    def create(
        cls,
        name: str,
        inputs: Dict[str, Union[Spec, PortSpec]] = None,
        outputs: Union[Spec, Dict[str, Union[Spec, PortSpec]]] = None,
        version: str = "1.0.0",
        pure: bool = False,
        stateful: bool = False,
        deterministic: bool = False,
        side_effects: List[str] = None,
        requires_models: List[str] = None,
        requires_buffers: List[str] = None,
        requires_optimizer: bool = False,
        allowed_contexts: Set[str] = None,
        tags: Set[str] = None,
        shape_fn: Optional[Callable[[Dict[str, Spec]], Dict[str, Spec]]] = None,
        estimated_flops: int = 0,
        memory_reads: int = 0,
        kernel_launch_cost: float = 0.0,
    ) -> "OperatorSpec":
        """Helper to create a spec with defaults and single output support."""
        # Process inputs: Wrap raw Specs in PortSpec
        processed_inputs = {}
        if inputs:
            for k, v in inputs.items():
                processed_inputs[k] = v if isinstance(v, PortSpec) else PortSpec(spec=v)

        # Process outputs: Wrap raw Specs in PortSpec
        processed_outputs = {}
        if outputs is not None:
            if not isinstance(outputs, dict):
                outputs = {"default": outputs}
            for k, v in outputs.items():
                processed_outputs[k] = (
                    v if isinstance(v, PortSpec) else PortSpec(spec=v)
                )

        return cls(
            name=name,
            version=version,
            inputs=processed_inputs,
            outputs=processed_outputs,
            pure=pure,
            stateful=stateful,
            deterministic=deterministic,
            side_effects=side_effects or [],
            requires_models=requires_models or [],
            requires_buffers=requires_buffers or [],
            requires_optimizer=requires_optimizer,
            allowed_contexts=allowed_contexts or {"actor", "learner"},
            tags=tags or set(),
            shape_fn=shape_fn,
            estimated_flops=estimated_flops,
            memory_reads=memory_reads,
            kernel_launch_cost=kernel_launch_cost,
        )


# Helper functions to create specs with a cleaner syntax
def Tensor(shape: Tuple[int, ...], dtype: str, rl_type: Optional[RLType] = None) -> TensorSpec:
    """Creates a TensorSpec with specified shape and dtype."""
    return TensorSpec(shape=shape, dtype=dtype, rl_type=rl_type)


def Scalar(dtype: str) -> TensorSpec:
    """Creates a 0-d TensorSpec (scalar) with specified dtype."""
    return TensorSpec(shape=(), dtype=dtype)


def Trajectory(length: Union[int, str]) -> TrajectoryType:
    return TrajectoryType(length=length)


def Episode() -> EpisodeType:
    return EpisodeType()


def Distribution(dist_type: str, is_logits: bool = False) -> DistributionType:
    return DistributionType(dist_type=dist_type, is_logits=is_logits)


def PolicySnapshot(version: int = 0) -> PolicySnapshotType:
    return PolicySnapshotType(version=version)


def ReplayBatch() -> ReplayBatchType:
    return ReplayBatchType()


def ScalarMetric() -> ScalarMetricType:
    return ScalarMetricType()


def RNGKey() -> RNGKeyType:
    return RNGKeyType()


def HiddenState() -> HiddenStateType:
    return HiddenStateType()


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


import warnings


def register_spec(node_type: str, spec: OperatorSpec) -> None:
    """Registers a specification for a given node type."""
    if node_type in _SPEC_REGISTRY:
        existing = _SPEC_REGISTRY[node_type]
        if existing == spec:
            return  # Identical, skip warning
        if existing.version == spec.version:
            warnings.warn(
                f"Duplicate version '{spec.version}' for operator '{node_type}'. "
                "Registering multiple times with the same version is discouraged."
            )
    _SPEC_REGISTRY[node_type] = spec


def get_spec(node_type: str) -> Optional[OperatorSpec]:
    """Retrieves the specification for a node type, or None if not registered."""
    return _SPEC_REGISTRY.get(node_type)


def clear_registry() -> None:
    """Resets the operator specification registry. Primarily used for testing."""
    _SPEC_REGISTRY.clear()


def is_compatible(src: Spec, dst: Spec) -> bool:
    """
    Checks if a source spec is compatible with a destination spec.

    Rules:
    - Both must be TensorSpecs or both must be Schemas.
    - If TensorSpecs: shapes and dtypes must match.
    - If Schemas: field sets, shapes, and dtypes must match.
    """
    if isinstance(src, TensorSpec) and isinstance(dst, TensorSpec):
        # Check semantic types if present
        if src.rl_type and dst.rl_type:
            if not src.rl_type.is_compatible(dst.rl_type):
                return False
        
        # We allow -1 to match any dimension for now (simple broadcast/flexible batch check)
        if len(src.shape) != len(dst.shape):
            return False
        for s, d in zip(src.shape, dst.shape):
            if s != -1 and d != -1 and s != d:
                return False
        return src.dtype == dst.dtype

    if isinstance(src, Schema) and isinstance(dst, Schema):
        return src.is_compatible(dst)

    # Check RLType objects directly
    if isinstance(src, RLType) and isinstance(dst, RLType):
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
