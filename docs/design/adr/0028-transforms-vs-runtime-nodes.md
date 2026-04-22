# ADR-0028: Separate Transforms from Runtime Nodes

## Status
Proposed

## Context
Initial iterations of the component model (ADR-0008) mixed pure mathematical logic (like computing an advantage) with runtime/scheduling concerns (like how that computation is dispatched or where its state is stored). This conflation made it difficult to reuse mathematical logic across different execution contexts (e.g., local execution vs. remote Ray actors).

## Options Considered

### Option 1: Everything Is One Component Type
- **Pros**
    - Single, simple mental model for the user.
    - Flat class hierarchy.
- **Cons**
    - **Conflated Concerns**: Pure math is buried inside boilerplate related to execution management.
    - **Reuse barriers**: Hard to take a "PPO Loss Component" and run it in a pure-functional JAX loop if it's tied to the `ExecutionEngine`'s lifecycle.

### Option 2: Transforms + Nodes (Chosen)
- **Pros**
    - **Clear Separation of Concerns**: Math lives in `Transforms`; execution boundaries live in `Nodes`.
    - **Easier Testing**: `Transforms` can be tested as pure functions with no engine overhead.
    - **Reusability**: The same `LossTransform` can be wrapped in a `LocalNode` for debugging or a `RemoteNode` for distributed training.
    - **Optimizer Opportunities**: The compiler can fuse multiple stateless `Transforms` into a single execution step.
- **Cons**
    - Introduces two concepts where there was previously one, requiring more documentation and user education.

## Decision
We propose adopting this approach because the system will formally separate **Transforms** from **Runtime Nodes**.

1. **Transforms**: These are stateless, functional units that perform pure data transformations (e.g., math, preprocessing). They define clear `keys_in` and `keys_out` but do not manage execution state, device placement, or distributed logic.
2. **Nodes**: These are the structural elements of the **Unified Graph IR (ADR-0026)**. A Node acts as a container for one or more Transforms and defines the execution context (e.g., "Run this transform on GPU 0," or "Run this transform asynchronously on a remote worker").

The Graph Engine will primarily work with Nodes, which in turn delegate to their encapsulated Transforms for the actual computation.

## Consequences

### Positive
- **Cleaner APIs**: Mathematical code is no longer cluttered with engine-specific hooks.
- **Kernel Fusion**: The compiler can identify chains of stateless Transforms and potentially fuse them into a single `torch.compile` region for maximum performance.
- **Polymorphism**: You can change the "Runtime Behavior" of a system (e.g., from synchronous to asynchronous) by swapping the outer `Nodes` without touching any of the mathematical `Transforms`.

### Negative / Tradeoffs
- **Complexity**: New developers must understand the difference between a mathematical operation (Transform) and the graph unit that executes it (Node).

## Notes
`Transforms` are analogous to functional operations (the "what"), while `Nodes` are the execution units of the schedule (the "where" and "how").