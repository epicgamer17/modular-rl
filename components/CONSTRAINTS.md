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

## 1. Contracts are semantic, not structural

Components communicate using **meaning**, not raw tensor shapes.

### ✅ Good
```python
requires = {"value": ValueEstimate}
❌ Bad
requires = {"value": Tensor[B, T, 51]}

Rationale:

Enables polymorphism (scalar, distributional, sequence)
Prevents combinatorial explosion of components
Decouples logic from representation
2. Every component must declare contracts

Each component must define:

requires = {...}
provides = {...}

This enables:

DAG construction
dependency validation
system introspection
3. Constraints describe, validation enforces

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

Each component may define:

def validate(bb):
    assert same_batch(bb["value"], bb["target"])
This is the source of truth for correctness
Use for:
polymorphic logic
conditional constraints
representation-specific checks
complex invariants
Example:
def validate(bb):
    value = bb["value"]
    target = bb["target"]

    assert same_batch(value, target)

    if is_distributional(value):
        assert value.num_atoms == target.num_atoms
    else:
        assert is_scalar(target)
Constraint Scope
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
Semantic Type System

All components operate on a shared set of semantic types.

Examples:
Observation
Action
PolicyLogits
ActionDistribution
ValueEstimate
ValueTarget
Reward
Trajectory
HiddenState
LossScalar

These types define meaning, not structure.

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