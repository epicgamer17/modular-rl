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

### ✅ Good
```python
@property
def requires(self) -> Set[Key]:
    return {Key("predictions.values", ValueEstimate)}
```
❌ Bad
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
        Key("predictions.values", ValueEstimate): "new",
        Key("meta.processed", SemanticType): "overwrite"
    }
```


This enables:

*   **DAG Validation**: Verifying data flow before anything runs.
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

Each component may define a `validate()` method. To ensure consistency and reduce duplication, components SHOULD use centralized validation helpers from `core.validation`.

### Validation Helpers
*   `assert_same_batch(t1, t2)`: Ensures dim 0 matches.
*   `assert_time_dim(t, expected)`: Ensures dim 1 matches search unroll or sequence length.
*   `assert_compatible_value(pred, target)`: Checks distributional vs scalar compatibility.

### Example
```python
from core.validation import assert_same_batch, assert_compatible_value

def validate(self, bb: Blackboard):
    pred = bb.predictions["values"]
    target = bb.targets["values"]

    assert_same_batch(pred, target)
    assert_compatible_value(pred, target)
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
class DiscreteValue(ValueTarget): pass
class ProjectValue(ValueTarget): pass
```

If a component requires `ValueTarget`, it will accept `DiscreteValue` automatically because the DAG validator uses `issubclass()`.

### The Key Object

A `Key` binds a **blackboard path** (string) to a **semantic type**.

```python
Key("targets.policies", PolicyLogits)
```

*   **Path**: Used by `resolve_blackboard_path` to find tensors.
*   **Type**: Used by the engine to validate DAG topology.

Adapters (Normalization Layer)

Adapters bridge representation differences.

Examples:
to_expectation(value)
ensure_time_dim(x)
align_batch(x, y)
Rules:
Components should rely on adapters instead of handling all cases manually
Adapters should be centralized and reusable
DAG Validation
Build-time validation

The system should:

verify all required inputs are provided
check type compatibility
optionally check declarative constraints
Runtime validation

In debug mode:

run validate() for each component
check invariants
detect NaNs or invalid values
Execution Modes
Strict Mode (Debugging)
run all validate() functions
enforce additional checks
slower but safer
Relaxed Mode (Training)
minimal validation
rely on adapters
optimized for performance
Design Guidelines
When to use declarative constraints

Use if:

the rule is simple
it improves readability
it helps tooling

Otherwise → use validate()

When to use programmatic validation

Use if:

logic depends on representation
behavior is conditional
correctness is critical
When to add adapters

Add adapters when:

multiple representations exist
components would otherwise duplicate logic
normalization simplifies downstream components
Anti-Patterns
❌ Constraint-only systems
insufficient for real RL complexity
cannot handle polymorphism
❌ Shape-driven design
leads to rigid systems
breaks composability
❌ Per-representation components
leads to explosion of variants
hard to maintain
❌ Hidden assumptions
components silently expecting specific formats
no validation or documentation
Summary

This system is built on three pillars:

1. Semantic contracts

Define what data means

2. Programmatic validation

Ensure correctness

3. Adapters

Handle representation differences

Key Insight

Components should agree on meaning, not representation.

Philosophy
Prefer flexibility over rigidity
Prefer explicit validation over implicit assumptions
Prefer composition over specialization
Prefer simple rules over complex DSLs

This approach enables:

modular RL systems
safe experimentation
scalable architecture
maintainable complexity