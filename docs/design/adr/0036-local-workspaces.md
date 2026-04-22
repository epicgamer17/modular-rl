# ADR-0036: Keep Workspace Local, Avoid Global Shared Blackboard

## Status
Proposed

## Context
In early iterations, the system relied on a single, shared "Blackboard" or "Workspace" to store all data for a training run. While simple, global mutable shared state creates significant architectural issues:
- **Tight Coupling**: Any component can read or write any value, making it impossible to reason about data flow without reading every line of code.
- **Ambiguity**: It is unclear which part of the system "owns" a specific tensor or when it was last updated.
- **Concurrency Hazards**: Parallel execution of independent graph branches becomes difficult if they all compete for the same global keys.
- **Testing Difficulties**: Unit testing a single component requires carefully setting up a global state that might contain hundreds of irrelevant keys.

## Options Considered

### Option 1: One Global Blackboard (Standard Shared Memory)
- **Pros**
    - Easy data access from anywhere in the system.
    - No need to define explicit data boundaries between subgraphs.
- **Cons**
    - Escalating complexity as the system grows.
    - Hard to debug data corruption bugs.

### Option 2: Local Workspaces Per Execution Scope (Chosen)
- **Pros**
    - **Isolation**: A graph run has its own local workspace that only contains the data relevant to that specific execution plan.
    - **Clear Ownership**: The inputs and outputs of a graph are explicitly mapped at the boundaries, eliminating "spooky action at a distance."
    - **Parallel Safety**: Independent subgraphs can run in their own local workspaces without any risk of interference.
- **Cons**
    - Requires explicit **Boundary Plumbing** (port mapping) to move data between parent and child workspaces (see ADR-0028).

## Decision
We propose adopting this approach because the system will utilize **Local Execution Workspaces** scoped to specific graph runs or composite nodes.

1. **Isolation**: Every execution of a Graph or Composite Node creates (or is assigned) a local Workspace.
2. **Key Visibility**: A component inside a local workspace can only see the keys provided to that workspace via its external ports or produced by its sibling components.
3. **No Global State**: There is no "Root Blackboard" that all components share by default.

## Consequences

### Positive
- **Cleaner Composition**: Nested graphs (ADR-0028) are truly encapsulated; they cannot accidentally overwrite keys in the parent graph.
- **Easier Testing**: A subgraph can be tested in total isolation by simply providing a small, local workspace with the required input keys.
- **Deterministic Execution**: The inputs required for a graph run are explicitly known at the start of the execution, as mandated by the **Graph IR (ADR-0026)**.

### Negative / Tradeoffs
- **Plumbing Overhead**: Passing a value through multiple levels of a hierarchical graph requires explicit port mappings (mappings of parent keys to child keys).

## Notes
A Workspace is treated as **Runtime Storage (Backend detail)**, not as an **Architectural Surface**. The architecture is defined by the Graph and its Contracts, while the Workspace is simply the memory where that data is temporarily held during execution.