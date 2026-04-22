# ADR-0014: Static Graph Validation Before Execution

## Status
Accepted

## Context
Executing Reinforcement Learning (RL) pipelines as dynamic Directed Acyclic Graphs (DAGs) introduces several failure modes that are difficult to debug if they occur at runtime:
- **Missing Dependencies**: A component required for a loss calculation isn't provided by any upstream node.
- **Circular Dependencies**: Components accidentally depend on each other's outputs.
- **Shape/Type Mismatches**: A component expects a `[B, T, D]` tensor but receives `[B*T, D]`.
- **Side-Effect Pruning Errors**: The system removes a component it deems "unnecessary," only to find later that it was required for a hidden logging side-effect.

Crashing hours into a training run because of a configuration error is a major bottleneck for research.

## Options Considered
### Option 1: Just-in-Time (JIT) Execution
- **Pros**
    - Flexible; allows for dynamic branch switching.
    - Simpler to implement initially (no compilation pass).
- **Cons**
    - Runtime errors are hard to trace.
    - Expensive experiments can fail late due to trivial configuration bugs.

### Option 2: Pre-Execution Graph Validation (Compilation)
- **Pros**
    - **Fail-Fast**: Errors are caught at startup, not mid-run.
    - **Optimization**: Allows for deterministic side-effect pruning and memory allocation planning.
    - **Self-Documenting**: The validation process can output the final execution order and dependency map.
- **Cons**
    - Requires a formal contract system (Semantic Types) to be implemented for every component.

## Decision
We will implement the system will perform a mandatory **Validation/Compilation Pass** before the execution loop begins.

During this pass, the `ExecutionEngine` (or `BlackboardEngine`) will:
1. **Resolve Dependencies**: Ensure every `key_in` required by a component is provided by a `key_out` earlier in the chain or by the initial data batch.
2. **Detect Cycles**: Verify the graph is a valid DAG.
3. **Verify Semantic Contracts**: Check that provided types match expected types (e.g., `ValueEstimate` vs `PolicyDist`).
4. **Static Pruning**: Determine exactly which nodes are required to produce the requested "terminal" keys and prune everything else.
5. **Shape Integrity**: (Where possible) Propagate tensor shapes through the graph to detect mismatches.

## Consequences
### Positive
- **Reliability**: Eliminates a large class of configuration-related runtime errors.
- **Transparency**: Developers can inspect the "vetted" execution plan before running it.
- **Efficiency**: Pruning unnecessary nodes reduces computation overhead.

### Negative / Tradeoffs
- **Rigidity**: Components must strictly declare their inputs and outputs up front; "magic keys" or late-binding attributes are forbidden.
- **Initial Complexity**: Implementing a robust validation engine requires significant upfront effort.

## Notes
This ADR builds upon **ADR-0004 (Contract-Driven Execution Graph)** and **ADR-0008 (Pipeline Components as DAG Nodes)**, moving from "we use a DAG" to "the DAG must be statically verified."