# ADR-0030: Use Multiple Executors Over Same Graph IR

## Status
Accepted

## Context
Different Reinforcement Learning (RL) workloads require vastly different runtime strategies. A local debugging session with a small model requires minimal overhead and high interactivity, while a large-scale training run requires complex asynchronous dispatching, distributed scheduling (e.g., via Ray), and hardware-specific optimizations.

Hardcoding the system to a single execution engine makes it difficult to support these diverse use cases without introducing significant complexity to the core computation logic.

## Options Considered

### Option 1: One Unified Executor
- **Pros**
    - Simpler initial implementation; no need for executor interfaces.
    - Guaranteed identical behavior across all runs.
- **Cons**
    - **Poor Flexibility**: The executor becomes a bottleneck of "lowest common denominator" features, or it becomes bloated with complex conditional logic for different environments.
    - **Debugging Friction**: Distributed executors are notoriously difficult to debug locally.

### Option 2: Multiple Executors (Chosen)
- **Pros**
    - **Precision**: Different executors can be optimized for specific environments (e.g., a `SequentialExecutor` for unit tests vs. a `RayExecutor` for production).
    - **Experimentation**: Researchers can swap executors to test new scheduling strategies without modifying the mathematical model.
    - **Backend Portability**: The same **Graph IR (ADR-0026)** can be targets for future backends (e.g., a hardware-accelerated C++ executor).
- **Cons**
    - Requires maintaining more backend-specific code.
    - Demands rigorous **semantic consistency tests** to ensure that different executors produce the same results for the same graph.

## Decision
 We will use the following: the system will support **Multiple Executors** all targeting the same **Unified Graph IR**.

The core architecture will decouple the *definition* of the execution plan (the Graph IR) from the *dispatching* of that plan (the Executor). 

Standard executors include:
1. **`SequentialExecutor`**: A simple, single-threaded executor ideal for local debugging and unit tests.
2. **`WorkspaceExecutor` (Blackboard-backed)**: An optimized local executor that utilizes a shared memory blackboard (see ADR-0027) for zero-copy data passing.
3. **`AsyncExecutor` / `DistributedExecutor`**: Future executors designed for asynchronous and remote node execution.

## Consequences

### Positive
- **Backend Portability**: The computation logic (Components/Transforms) is completely decoupled from how it is scheduled and run.
- **Improved Developer Experience**: Developers can step through their RL logic in a `SequentialExecutor` with standard debuggers before deploying to a distributed cluster.
- **Performance Tuning**: Executors can implement different optimization strategies (e.g., parallelizing independent branches) independently of the compute nodes.

### Negative / Tradeoffs
- **Complexity**: Requires a well-defined `Executor` interface and shared telemetry/error-handling protocols.
- **Testing Burden**: Ensuring that a graph runs identically on all supported executors is a significant verification challenge.

## Notes
Examples of proposed executors include `SequentialExecutor`, `WorkspaceExecutor`, and potentially a JIT-compiled executor.