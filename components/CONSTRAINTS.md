# Constraint & Validation System (DAG Philosophy)

## Overview

This system defines how components declare, validate, and enforce their data requirements within a DAG-based RL architecture.

The core goal is to balance:

- **Flexibility** (support many representations, algorithms, and compositions)
- **Correctness** (prevent invalid or inconsistent configurations)
- **Clarity** (make systems understandable and debuggable)

This is achieved through a **layered contract system**:
1. Semantic contracts (required)
2. Declarative constraints (optional, lightweight)
3. Programmatic validation (required for enforcement)

---

# Core Principles

## 3. Contracts must be deterministic after initialization

To prevent "shifting sand" bugs during execution, a component's contract (what it reads and writes) MUST NOT change after the component has been initialized.

*   **DO**: Compute `requires` and `provides` once in `__init__`.
*   **AVOID**: Logical branching inside the `@property` methods.
*   **FIXED**: Contracts are declared once, enabling static DAG validation before the first training step.

### Example
```python
class MyComponent(PipelineComponent):
    def __init__(self, mode: str):
        # Compute contract once based on initialization parameters
        base_key = "data.x" if mode == "fast" else "data.y"
        self._requires = {Key(base_key, Observation)}
        self._provides = {Key("targets.z", Observation)}

    @property
    def requires(self) -> Set[Key]:
        return self._requires
    
    @property
    def provides(self) -> Set[Key]:
        return self._provides
```

## 4. Contracts are semantic, not structural

Components communicate using **meaningful keys** and **semantic types**, not raw strings or tensor shapes.

### ✅ Good (Parameterized)
```python
@property
def requires(self) -> Set[Key]:
    return {Key("predictions.values", ValueEstimate[Scalar])}
```

### ✅ Good (Structural)
```python
@property
def provides(self) -> Dict[Key, str]:
    # Returns ValueEstimate parameterized with Categorical structure
    return {Key("predictions.values", ValueEstimate[Categorical(bins=51)]): "new"}
```

❌ Bad (Generic Strings)
```python
requires = {"value": "tensor"}
```

Rationale:

*   **Type Safety**: Prevents passing a `PolicyLogits` tensor into a `ValueLoss` component.
*   **Polymorphism**: Enables components to accept specific types (e.g., `DiscreteValue`) where a base type (`ValueEstimate`) is expected.
*   **Decoupling**: Logic is isolated from specific blackboard path naming conventions via configurable keys.

## 5. Every component must declare bound contracts

Contracts MUST be instance properties (`@property`), never class-level attributes. This allows components to be dynamic based on their configuration (e.g., different target keys).

Each component MUST define:

*   `requires` -> `Set[Key]`
*   `provides` -> `Dict[Key, str]`

### Write Modes (`provides`)
Components communicate their intent for writing data using write modes. The `provides` method SHOULD return a dictionary mapping `Key` to a `WriteMode` string.

*   `"new"`: (Default) Expects the key to NOT exist. Used for first-time production of data.
*   `"overwrite"`: Expects the key to exist. Explicitly signals that a downstream component is modifying an upstream value (e.g., gradient scaling).
*   `"append"`: For collections (lists/dicts) where data is added rather than replaced.
*   `"optional"`: Signals that the key MAY be produced depending on internal logic.

### Example:
```python
def provides(self) -> Dict[Key, str]:
    return {
        Key("predictions.values", ValueEstimate[Scalar]): "new",
        Key("meta.processed", SemanticType): "overwrite"
    }
```


This enables:

*   **DAG Validation**: Verifying data flow and representation consistency before anything runs.
*   **Automated Discovery**: Eliminates combinatorial overhead by querying `agent_network.get_learner_contract()` instead of passing distribution strings.
*   **Type Matching**: Using `issubclass(found_type, required_type)` to allow polymorphic data flow.
*   **Path Resolution**: Mapping complex paths like `targets.policies` to semantic identifiers.

We separate:

Layer	Purpose
constraints	human-readable description
validate()	actual enforcement
Constraint Types
1. Declarative Constraints (Optional)

Simple, human-readable relationships:

constraints = [
    "same_batch(value, target)",
    "value.time_dim in {None, T}"
]
Use cases:
DAG visualization
config validation
documentation
Rules:
Must be simple (single-line logic)
No branching or complex conditions
No deep type inspection
Anti-pattern:

If a constraint requires explanation, it belongs in validate().

2. Programmatic Validation (Required)

Each component MUST define a `validate()` method that enforces its data assumptions. To ensure consistency and reduce duplication, components SHOULD use centralized validation helpers from `core.validation`.

### Validation Principles
1. **Robust Checks**: Don't just check for key existence. Check types (`assert_is_tensor`), rank (`assert_shape_sanity`), and compatibility.
2. **Delegate to Strategy**: If a component uses a `BaseRepresentation` or similar strategy object, delegate the validation of specific math invariants to that object (e.g., `rep.validate_logits(tensor)`).
3. **Source of Truth**: `validate()` should be the single source of truth for correctness. `execute()` should be lean and trust the validation layer.

### Validation Helpers
*   `assert_in_blackboard(bb, key)`: Verifies key or path existence.
*   `assert_is_tensor(obj)`: Ensures object is a PyTorch tensor.
*   `assert_shape_sanity(t, min_ndim, max_ndim)`: Verifies tensor rank (B, T, D alignment).
*   `assert_same_batch(t1, t2)`: Ensures dim 0 matches across tensors.
*   `assert_compatible_value(pred, target)`: Verifies distributional vs scalar compatibility.
*   `assert_representation_supports(rep, tensor)`: Delegates detailed math validation to a representation strategy.

### Example
```python
from core.validation import assert_in_blackboard, assert_is_tensor, assert_representation_supports

def validate(self, bb: Blackboard):
    assert_in_blackboard(bb, f"predictions.{self._logits_key}")
    logits = bb.predictions[self._logits_key]
    assert_is_tensor(logits)
    assert_representation_supports(self._representation, logits)
```

### Explicit vs. Implicit Mutations (`execute`)
While components MAY mutate the blackboard in-place for performance or complexity reasons, they SHOULD return a dictionary of their primary outputs.

*   **Implicit (In-place)**: `blackboard.losses["v"] = loss`. Flexible but hard to trace.
*   **Explicit (Return)**: `return {"losses.v": loss}`. Enables the framework to log, trace, and validate every change.

Components that return their changes are easier to debug and test in isolation.

Example:
```python
def execute(self, blackboard) -> Dict[str, Any]:
    loss = self.compute(blackboard)
    return {f"losses.{self.name}": loss}
```

This is the source of truth for correctness. Use it for representation-specific checks and complex invariants.
What constraints SHOULD enforce
1. Relationships between data
batch alignment
time dimension compatibility
shape compatibility (relative, not absolute)
2. Representation compatibility
scalar vs distributional
logits vs probabilities
3. Algorithmic invariants
required inputs exist
outputs are well-formed
What constraints SHOULD NOT enforce
❌ Exact tensor shapes (globally)
breaks composability
reduces flexibility
❌ Implementation details
device placement
internal network structure
❌ Redundant checks
if adapters already normalize inputs

### Semantic Type System

All components operate on a shared hierarchy of `SemanticType` classes.

### Core Types
*   `Observation`
*   `Action`
*   `PolicyLogits`
*   `ActionDistribution`
*   `ValueEstimate`
*   `ValueTarget`
*   `Reward`
*   `LossScalar`
*   `ToPlay`

### Type Inheritance (Polymorphism)

Systems often require specific data representations that still share a high-level meaning. We use standard Python inheritance to model this:

```python
class ValueTarget(SemanticType): pass
class DiscreteValue(ValueTarget[Categorical(bins=51)]): pass
class ScalarValue(ValueTarget[Scalar]): pass
```

If a component requires `ValueTarget`, it will accept `DiscreteValue` automatically because the DAG validator uses `issubclass()`.

### The Key Object

A `Key` binds a **blackboard path** (string) to a **semantic type**, and optionally, **metadata**.

```python
Key("targets.values", ValueTarget[Categorical(bins=51)])
```

*   **Path**: Used by `resolve_blackboard_path` to find tensors.
*   **Type**: Used by the engine to validate DAG topology.
*   **Metadata**: Used to enforce cross-component invariants (e.g. matching bin counts).

---

# 6. Global Pipeline Optimization & Execution Graph

To ensure high-performance execution and global correctness, the system builds an explicit **Execution Graph** before the first training step.

## Build-time DAG Validation
The `BlackboardEngine` enforces contract consistency before the first training step via a **4-stage validation pipeline**:

1.  **Dependency Resolution**: Ensures every `requires` path is produced by an earlier component or initial key.
2.  **Semantic Compatibility**: Verifies that the provided `SemanticType` satisfies the consumer's requirement via `issubclass()` (e.g., `ValueEstimate[Categorical] ⊆ ValueEstimate`).
3.  **Representation Consistency**: Uses `representation.get_metadata()` to ensure that providers and consumers agree on internal parameters (e.g., matching `vmin`, `vmax`, `bins`, or `num_classes`).
4.  **Shape Integrity**: Lightweight validation of dimensionality (rank) and structural properties (e.g., `has_time=True`).

### Rules for Optimal Pipelines:
*   **Never Hardcode Contracts**: Components should discover their contracts from the agent network via `get_learner_contract()` where possible.
*   **Avoid Hidden Dependencies**: Ensure every input is declared in `requires`. The pruner will remove components if it thinks their outputs aren't needed!
*   **Declare Metadata**: If your logic depends on a specific configuration (e.g. 51 bins), ensure it's captured in the `Key.metadata` via representations.
*   **Explicit Telemetry**: Ensure any component intended for logging writes to the `meta.` or `losses.` paths, as these are never pruned.

## Runtime Pipeline Safety
1. **Strict Mode Validation**: If `engine.strict=True`, `validate()` is called every step to catch dynamic shape or value drift.
2. **Deterministic Layouts**: Forward passes optimize memory layout (e.g., `channels_last`) based on discovered hardware support, transparent to the component logic.

IN FUTURE: 
1.  **Terminal-Value Pruning**: The engine automatically identifies and skips components whose outputs are never consumed by downstream components or terminal sinks (losses, telemetry).
2.  **Metadata Validation**: The engine enforces that all providers and consumers agree on semantic metadata (like `bins` or `support_range`), preventing subtle algorithmic bugs.
3.  **Future Proofing**: The explicit graph enables future optimizations like **component fusion** (combining small mathematical transforms) and **batching optimization**.
4. **Adaptive Execution**: The engine can skip branches of the DAG based on dynamic valves (e.g. `stop_execution`).
5. **Transparent Dataflow**: Every mutation is explicit, allowing per-step tracing and bottleneck analysis.
