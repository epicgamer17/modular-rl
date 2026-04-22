# ADR-0044: Explicit Return-Based Dataflow

## Status
Accepted

## Context
In most RL frameworks, components (like network layers or loss functions) often mutate a shared state object in-place. While flexible, this makes it difficult to:
- Trace where a specific value was created or modified.
- Automatically log or version data changes.
- Ensure that components only modify what they declared they would provide.

## Options Considered

### Option 1: Implicit In-place Mutation
- **Pros**: Standard practice in many libraries (e.g., PyTorch modules mutating a `dict`). High performance for large tensors.
- **Cons**: Opaque; requires manual tracking of state changes. Hard to debug "where did this NaN come from?"

### Option 2: Explicit Return Mapping (Chosen)
- **Pros**: Enables the framework to intercept, validate, and log all changes. Makes components "purer" and easier to test.
- **Cons**: Small memory overhead for creating return dictionaries.

## Decision
Components SHOULD prefer returning a dictionary of their primary outputs rather than mutating the Blackboard in-place.

1. **The `execute()` Return**: The `execute(blackboard)` method should return a `Dict[str, Any]` (or `Dict[Key, Any]`) mapping blackboard paths to new result tensors.
2. **Framework Integration**: The `ExecutionEngine` is responsible for taking these returned values and merging them back into the Blackboard, performing safety checks (e.g., verifying the `WriteMode`) during the merge.
3. **Exceptions (Implicit Mutation)**: In-place mutation is still permitted for extremely complex or performance-critical operations (e.g., updating a massive replay buffer), but these must be clearly documented.

## Consequences

### Positive
- **Observability**: Every data change is a discrete event that can be logged or traced.
- **Isolation**: Components don't need to know the internal structure of the Blackboard, only their specific input/output keys.
- **Testability**: Unit tests can simply call `execute()` and assert on the returned dictionary without inspecting a shared state object.

### Negative / Tradeoffs
- **Boilerplate**: Components must explicitly construct and return a dictionary.

## Notes
Refines the "Explicit vs. Implicit Mutations" section of the Component Constraints.
