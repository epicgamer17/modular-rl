# ADR-0042: Layered Validation (Declarative vs Programmatic)

## Status
Accepted

## Context
Validating RL components involves checking multiple layers of correctness:
1. **Structural**: Are the keys there?
2. **Semantic**: Is it the right type of data (e.g., Logits vs Probs)?
3. **Mathematical**: Do the shapes align? Are the values in range?

A single `validate()` method for all components often leads to code duplication and makes it hard for external tools to understand the component's assumptions without executing code.

## Options Considered

### Option 1: Pure Programmatic Validation
- **Pros**: Maximum power; can check anything.
- **Cons**: Opaque to static analysis; leads to duplicate logic for common checks (e.g., `assert_same_batch`).

### Option 2: Pure Declarative Constraints
- **Pros**: Transparent; can be used for auto-documentation and visualization.
- **Cons**: Hard to express complex math invariants (e.g., "this tensor must support the current categorical distribution strategy").

### Option 3: Layered Validation (Chosen)
- **Pros**: Best of both worlds; light declarative metadata for tools, heavy programmatic enforcement for correctness.
- **Cons**: Slightly more boilerplate to define both.

## Decision
We will adopt a **layered validation strategy** that separates human-readable constraints from programmatic enforcement.

1. **Declarative Constraints (Optional)**: A list of simple, single-line strings or objects that describe relationships between data (e.g., `"same_batch(value, target)"`). These are used for documentation and lightweight configuration checks.
2. **Programmatic Validation (Required)**: A mandatory `validate(blackboard)` method that performs robust checks.
   - **Centralized Helpers**: Use `core.validation` helpers (e.g., `assert_shape_sanity`) to ensure consistency.
   - **Delegation**: If a component uses a strategy object (e.g., `BaseRepresentation`), it must delegate math-specific validation to that object.
   - **Single Source of Truth**: `validate()` is the final word on correctness, allowing `execute()` to remain lean and performant.

## Consequences

### Positive
- **Fail Fast**: Catch complex algorithmic bugs before the first tensor operation.
- **Consistency**: Centralized helpers ensure that "batch alignment" is checked the same way across all components.
- **Tooling Support**: Declarative constraints can be parsed to generate interactive DAG visualizations.

### Negative / Tradeoffs
- **Execution Overhead**: Substantial validation logic can slow down training if not toggled off in production (see `engine.strict` flag).

## Notes
Consolidates the "Constraint Types" and "Validation Principles" sections of the Component Constraints.
