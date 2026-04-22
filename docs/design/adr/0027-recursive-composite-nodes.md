# ADR-0027: Use Recursive Composite Nodes for Hierarchical Graphs

## Status
Proposed (Partial Implementation)

## Context
High-level systems (such as a `FullLearner` or a `DistributedActor`) frequently consist of multiple internal components organized in a specific workflow. These complex systems must be able to expose their internal logic as a graph while still functioning as a single unit or "Node" when viewed from an even higher layer (the **Macro layer**, see ADR-0026).

## Options Considered

### Option 1: Special Wrapper Nodes
- **Pros**
    - Quick to implement using standard object-oriented wrappers/proxies.
- **Cons**
    - Creates "split semantics" where internal logic is a graph but externally it is an opaque function.
    - Prevents global optimization passes (like pruning) from seeing through the wrapper into the internal execution chain.

### Option 2: Composite Nodes as Nested Graphs (Chosen)
- **Pros**
    - **Recursive Structure**: The same semantic model (Unified Graph IR, ADR-0026) applies at every level of depth.
    - **Uniform Contracts**: A Composite Node's external contract (inputs/outputs) is simply the union of the unresolved boundaries of its internal graph.
    - **Full Transparency**: A top-level compiler can "flatten" nested graphs during optimization to perform system-wide dependency analysis.
- **Cons**
    - Requires formal "Port" or "Boundary" definitions to map external context keys to internal subgraph keys.

## Decision
We propose adopting this approach because hierarchical complexity will be handled via **Recursive Composite Nodes**. 

A `CompositeNode` is defined as a node whose internal implementation is itself a Graph of the same IR type. This allows the system to build arbitrary levels of hierarchy (e.g., a `TrainingStep` graph containing a `LossGraph` containing a `NetworkGraph`) without ever changing the fundamental unit of execution.

## Consequences

### Positive
- **Natural Hierarchy**: Reflects the way RL researchers think (e.g., "The PPO Learner contains an Advantage module and a Policy module").
- **Semantic Consistency**: Tools developed for simple nodes (validation, visualization, tracing) automatically work for entire nested systems.
- **Encapsulation with Visibility**: Complex systems can hide their internal implementation details behind a clean boundary port without sacrificing the metadata needed for cross-node optimization.

### Negative / Tradeoffs
- **Introspection Complexity**: Navigating a deep stack of nested graphs for debugging requires more sophisticated tooling than a flat list of components.

## Notes
A `CompositeNode` is a formal structural element, not a hidden adapter or a convenience wrapper. It must adhere to the same contract-driven requirements as any other node in the system.