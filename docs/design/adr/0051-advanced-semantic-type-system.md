# ADR-0051: Advanced Semantic Type System and Shape Inference

## Status
Proposed

## Context
Current contracts use simple semantic types. While helpful, they don't capture the full mathematical intent of the data (e.g., bounds, specific distributions). Furthermore, shape propagation across complex graph branches is often manual and error-prone.

## Options Considered

### Option 1: Basic Typing
- **Pros**: Low overhead; simple implementation.
- **Cons**: Leaves too many mathematical invariants ("is this hidden state normalized?", "is this a scalar or distribution?") to runtime checks.

### Option 2: Rich, Propagating Type System (Chosen)
- **Pros**: Enables static verification of mathematical compatibility and automated shape handling.
- **Cons**: Increases the complexity of the core contract library.

## Decision
We propose adopting a rich, propogating semantic type system that explicitly models mathematical intent.

1. **Rich Typology**: Introduce detailed types including `Box[low, high]`, `Discrete`, `Continuous`, `Trajectory[T]`, `NetworkParams[Type]`, and `Return[gamma]`.
2. **Bounds Validation**: Support `low`/`high` bounds as part of the `Box` and `Discrete` types (aligned with Gym standards) to automatically verify normalization and hidden state scales.
3. **Automated Propagation**: Implement shape and type inference across the execution graph. Ensure that if a component produces a `Distribution` (e.g., C51 value), it can be automatically converted to its `ExpectedValue` (Scalar) when consumed by a node expecting a scalar.
4. **Semantic Compatibility**: Enable the framework to bridge type mismatches automatically using semantic rules (e.g., auto-projecting distributions to scalars).

## Consequences

### Positive
- **Correctness**: Catches normalization or distribution-matching errors before execution.
- **Flexibility**: Simplifies swapping between different value representations (Scalar vs C51).
- **Automation**: Reduces the need for manual "formatter" components.

### Negative / Tradeoffs
- **Internal Complexity**: The Graph Compiler must now understand type transformation and propagation rules.
