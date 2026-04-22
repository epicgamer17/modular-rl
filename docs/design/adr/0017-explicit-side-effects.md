# ADR-0017: Explicit Declaration of Side Effects

## Status
Accepted

## Context
As defined in **ADR-0017 (Components Are Pure Dataflow Units)**, the system treats components as predictable math blocks for the purpose of graph optimization and pruning. However, a significant class of components exists whose primary purpose is an operational "action" rather than a data transformation:
- **Optimizer Step**: Updates network parameters.
- **Metrics Sink**: Sends data to an external logger.
- **Replay Append**: Pushes data to a storage buffer.
- **Checkpoint Save**: Writes the model to disk.

A standard Directed Acyclic Graph (DAG) pruner identifies "dead code" by checking if a node's outputs are used by any subsequent node. Since many of these operational components produce no Blackboard outputs, they are at high risk of being accidentally pruned during graph compilation.

## Options Considered
### Option 1: Implicit Side Effects (Unsafe)
- **Pros**
    - Less developer effort; "just works" if pruning is disabled.
- **Cons**
    - Any attempt at graph optimization (as described in ADR-0015) will break the system by removing critical nodes like the optimizer or logger.

### Option 2: Explicit Side Effect Declaration (Chosen)
- **Pros**
    - **Safe Pruning**: The compilation engine now knows which nodes are "terminal" markers that must be preserved regardless of usage.
    - **Auditability**: It is easy to generate a list of all "actions" a specific execution plan will perform.
    - **Optimization**: The system can intelligently group side-effect nodes (e.g., batching multiple logging calls).
- **Cons**
    - Requires developers to remember to set a `side_effects=True` flag or similar metadata.

## Decision
We will implement all components performing operations outside of writing to the Blackboard must **explicitly declare their side effects**.

This is enforced via:
1. **Metadata Flag**: A `side_effects=True` Boolean on the component class or instance.
2. **Lifecycle Hooks**: Registration for specific engine phases (e.g., `on_batch_end`) which are intrinsically protected from pruning.

Any node marked as having side effects is treated as a "Root of Interest" during the pruning pass, ensuring that it and all of its transitive dependencies (the keys it needs to perform its side effect) are preserved in the final execution graph.

## Consequences
### Positive
- **Reliability**: Eliminates the "invisible optimizer" bug where a training loop runs but parameters never update.
- **Correctness**: The graph pruner now has a complete understanding of what is "important" in the execution chain.
- **Performance**: Nodes that *honestly* have no side effects and no downstream consumers are still safely pruned, keeping the pipeline lean.

### Negative / Tradeoffs
- **Registration Overhead**: Small amount of extra boilerplate when creating new operational components.

## Notes
This ADR resolves a specific bug encountered during the transition to a contract-driven DAG system where the optimizer was being pruned because its "output" (updated weights) was not explicitly tracked as a Blackboard key.