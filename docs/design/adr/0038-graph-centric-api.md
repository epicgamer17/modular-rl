# ADR-0038: Keep User-Facing API Graph-Centric, Hide Execution Memory

## Status
Proposed

## Context
As the architectural foundation has evolved (ADR-0026 through ADR-0038), the system now has a clear distinction between the high-level **Graph IR** (what the user builds) and the low-level **Execution Backends** (how the graph is run, including Workspaces and Blackboards).

To ensure the framework remains accessible and maintainable, it is critical to decide which of these layers are exposed to the user. Exposing low-level memory internals (like the Blackboard) as a primary interface creates unnecessary complexity for the end user and leaks implementation details that restrict the system's ability to evolve its backend.

## Options Considered

### Option 1: Expose Blackboard Keys as the Primary API
- **Pros**
    - Provides power-users with absolute, low-level control over every tensor in the system.
    - Highly flexible for ad-hoc debugging and manual state mutation.
- **Cons**
    - **Leaky Abstraction**: Users must understand "Memory Workspaces" even if they only want to create a simple PPO trainer.
    - **Brittle Code**: User scripts become tightly coupled to internal key names and memory layouts (ADR-0037).

### Option 2: Graph-Centric API (Chosen)
- **Pros**
    - **Cleaner UX**: Users interact with logical "Nodes," "Graphs," and "Ports" (ADR-0028). The dataflow is expressed through connections and contracts, not global memory lookups.
    - **Better Portability**: A user-defined graph can be moved between process boundaries or different executors (ADR-0031) without changing a single line of user code.
    - **Abstraction Safety**: Implementation details (like the legacy Blackboard backend, ADR-0035) can be hidden or replaced without breaking user-facing scripts.
- **Cons**
    - Advanced debugging workflows may require specialized "escape hatches" or diagnostic nodes to inspect the internal memory state.

## Decision
We propose adopting this approach because the primary user-facing API of the system will be **strictly Graph-Centric**.

1. **User Surface**: Users will reason about and interact with `Node`, `Graph`, `Contract`, `Port`, and `Transform` objects.
2. **Hidden Internals**: The `Blackboard` / `Workspace` / `ExecutionState` models are strictly internal to the **Executor Backends** and will not be mentioned in standard user-facing tutorials or high-level trainer APIs.
3. **Data Access**: If a user needs to export data from a graph run, they must define an explicit **Terminal Port** or use a **Metric Sink Node** (ADR-0025), rather than directly querying an internal memory buffer.

## Consequences

### Positive
- **Cleaner Public Surface**: The documentation can focus on "how to build a graph," making the framework significantly easier to learn.
- **Improved Maintainability**: The internal "machine room" (ADR-0027) can be refactored or rewritten without impacting the user's "lobby" experience.
- **Architectural Purity**: Reinforces the **Pure Dataflow (ADR-0017)** and **Contract-Driven (ADR-0016)** nature of the system.

### Negative / Tradeoffs
- **Diagnostic Overhead**: Requires building dedicated expert/debug interfaces (e.g., a "Blackboard Visualizer" or "Contract Trace") to provide the transparency previously available through direct memory access.

## Notes
A well-designed framework should feel like drawing a logic diagram, not like managing a global memory pool. Users should not need deep knowledge of the Blackboard to perform normal research tasks.