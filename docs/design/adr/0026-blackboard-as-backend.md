# ADR-0026: Treat Blackboard as Executor Backend, Not Core Architecture

## Status
Proposed

## Context
The system has relied heavily on a Blackboard (shared context) model for data sharing between components. While highly efficient for local execution, elevating the Blackboard to a "core architecture" principle conflicted with the move toward explicit **Unified Graph IR (ADR-0026)** and contract-driven dataflow. Treating the Blackboard as the primary system model made remote execution (where no shared memory exists) conceptually difficult.

## Options Considered

### Option 1: Keep Blackboard as Primary System Model
- **Pros**
    - Reuses the current highly-optimized implementation without changes.
    - Provides high runtime flexibility for local execution.
- **Cons**
    - **Implicit Dataflow**: Dependencies are hidden inside the Blackboard, making it harder to reason about the graph.
    - **Composition Hurdles**: Harder to compose and transform graphs when data state is global to the execution context.
    - **Remote Inefficiency**: Forces remote nodes to "emulate" a blackboard, which is conceptually messy.

### Option 2: Demote Blackboard to Execution Backend (Chosen)
- **Pros**
    - **Performance**: Retains the zero-copy performance benefits of the Blackboard for local execution.
    - **Architecture Cleanliness**: Removes the conceptual split between "The Graph" and "The Blackboard."
    - **Flexible Backends**: Allows for different execution backends (e.g., a pure message-passing backend for distributed nodes) to satisfy the same Graph IR.
- **Cons**
    - Requires establishing formal Backend Interfaces for the Graph Engine.
    - Migration complexity when decoupling components from the specific Blackboard API.

## Decision
We propose adopting this approach because the **Blackboard (Workspace)** will be demoted from a core architectural pillar to a specific **Execution Backend** for the Graph IR.

In this model:
1. **Core Architecture**: Is defined strictly by the Graph IR and the explicit edges between nodes.
2. **Execution Backend**: The Graph Engine uses the Blackboard as its "scratch space" during local execution to store intermediate tensors, satisfying the requirements of the nodes.
3. **Backend Abstraction**: Components interact with a generic data accessor which, for local training, happens to be backed by a Blackboard.

## Consequences

### Positive
- **Valuable Legacy**: The existing, high-performance Blackboard engine remains a critical part of the localized "machine room."
- **Cleaner APIs**: High-level orchestration no longer needs to reason about Blackboard state, only about graph topology and data contracts.
- **Remote Transparency**: Distributed components see the same dataflow contracts without needing a global shared memory model.

### Negative / Tradeoffs
- **Abstractions**: Requires well-defined interfaces between the Graph Resolver and the data storage backends.
- **Migration**: Existing components that rely on "magic" Blackboard features (e.g., iterating over all keys) must be refactored to use explicit contracts.

## Notes
The Blackboard becomes the "machine room" (optimized for execution), not the "lobby" (where you design the architecture).