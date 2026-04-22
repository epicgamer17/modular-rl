# ADR-0032: Replay Buffer as Resource Node, Not Components

## Status
Proposed

## Context
Modeling Replay Buffers as standard stateless components within a dataflow graph (ADR-0017) proved to be awkward. Standard components are designed for high-frequency, transient data transformations. A Replay Buffer, however, is a persistent resource that manages its own memory, eviction policies, and sampling logic. 

Attempting to "force" the buffer into a stateless component required hiding its state in global variables or passing around large, opaque buffer handles through the execution graph, which violated the transparency principles of the system.

## Options Considered

### Option 1: ReplayStoreComponent / ReplaySampleComponent
- **Pros**
    - Familiar style; fits naturally into a flat list of components.
- **Cons**
    - **State Fragmented**: The state (the actual stored tensors) is hidden awkwardly behind the component instance.
    - **API Fragmentation**: Operations like `insert`, `sample`, and `update_priorities` must be implemented as separate components with different contracts, leading to potential out-of-sync bugs.

### Option 2: ReplayNode Resource (Chosen)
- **Pros**
    - **Explicit State Ownership**: The `ReplayNode` is a first-class **Service Node (ADR-0030)** that explicitly owns the lifecycle of the stored data.
    - **Clear Operations**: The node exposes a stable, well-defined set of operational methods (`insert`, `sample`, `update`).
    - **Unified Backend**: All storage logic is consolidated into a single architectural unit.
- **Cons**
    - Requires the introduction of **Service Semantics** into the execution engine to handle requests and replies.

## Decision
We propose adopting this approach because the Replay Buffer will be implemented as a **Resource Node (Service Node)**, exposing explicit `insert()`, `sample()`, and `update()` methods.

Key structural requirements:
1. **Method-Based Interaction**: Instead of just running a `forward()` pass, other nodes in the system (like the `Learner`'s sampling node) send specific requests to the `ReplayNode`.
2. **Pluggable Policies**: The core `ReplayNode` manages the "plumbing" (storage, serialization), but its behavior (e.g., Priority sampling, FIFO eviction, Reservoir sampling) is configured via **Pluggable Policies**.
3. **Blackboard-Driven IO**: While the internal storage is private, the inputs to `insert()` and the outputs from `sample()` are still standard **Blackboard Keys (ADR-0019)** governed by **Semantic Contracts (ADR-0016)**.

## Consequences

### Positive
- **Cleaner Mental Model**: Researchers treat the Replay Buffer as a service they interact with, rather than a mathematical block they pipe data through.
- **Extensibility**: It is significantly easier to implement advanced buffers (like Prioritized Experience Replay - PER) by simply swapping the internal policy without changing the Learner's execution graph.
- **Distributed Ready**: The `ReplayNode` can be easily moved to a remote process (e.g., a Ray Actor) because its interface is already based on asynchronous request/reply semantics.

### Negative / Tradeoffs
- **Policy Design**: Requires a robust internal policy API to prevent a proliferation of specialized `ReplayNode` classes.

## Notes
Policies should configure node behavior; we avoid creating many specialized replay node classes (e.g., avoid `PrioritizedReplayNode`, use `ReplayNode(policy=PrioritizedPolicy)`).