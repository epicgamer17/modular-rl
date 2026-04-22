# ADR-0039: Promote Existing DAG Compiler Into Global Graph Compiler

## Status
Proposed

## Context
As architectural layers were introduced—spanning from local **Transforms (ADR-0029)** to **Unified Graph IR (ADR-0026)** and **Composite Nodes (ADR-0028)**—the logic for dependency analysis, static validation, and graph pruning has remained largely confined to local "engine" implementations. 

Currently, the system's "compiler" logic is fragmented: the micro-layer (local execution) has strong validation, but the macro-layer (distributing actors and learners) relies on manual orchestration. To achieve the full potential of a **Derived Execution Graph (ADR-0024)**, the compiler must be a standalone, high-level service that can reason about the entire system end-to-end.

## Options Considered

### Option 1: Keep Compiler Logic Local-Only
- **Pros**
    - Less upfront structural refactoring.
    - Simpler local execution engine.
- Cons
    - **Duplicate Logic**: Macro-layer validation (connecting remote actors) would eventually require reimplementing the same dependency analysis.
    - **Optimization Barriers**: Pruning a node in a remote actor because its output isn't needed by the central learner is impossible if the compiler only sees the actor's local graph.

### Option 2: Promote to Global Graph Compiler (Chosen)
- **Pros**
    - **End-to-End Optimization**: Allows for cross-layer reasoning (e.g., pruning unnecessary observation preprocessing if those observations are never used for targets).
    - **Unified Pipeline**: A single optimization pipeline handles both local kernel fusion and remote distributed scheduling.
    - **Cleaner Architecture**: Centralizes the "intelligence" of the system into a single place, leaving the executors (ADR-0031) as thin, passive runners.
- **Cons**
    - **Significant Engineering Effort**: Requires a large-scale refactor to pull validation logic out of the executors and into a standalone compiler service.

## Decision
We propose adopting this approach because the planned architectural direction is to **evolve the local DAG compiler into a Universal Graph Compiler**.

This global compiler will:
1. **Analyze System-Wide Dependencies**: Not just within a single learner, but across all actors, buffers, and learners in a session.
2. **Handle Recursive Flattening**: Using the **Composite Node (ADR-0028)** structure to "see through" hierarchical boundaries.
3. **Generate Optimized Execution Artifacts**: Produce specialized instructions for different **Executors (ADR-0031)** based on the global analysis.

## Consequences

### Positive
- **Architectural Scaling**: The system remains lean and fast even as it scales to hundreds of distributed nodes.
- **Improved Performance**: Global pruning and kernel fusion provide speedups that are mathematically derived rather than manually tuned.
- **Strong Safety**: A single "Contract Validator" ensures that every data point in the entire distributed system is semantic and shape-compliant.

### Negative / Tradeoffs
- **Large Implementation Scope**: This represents a significant engineering milestone that will take time to fully realize.

## Notes
This is likely a later-phase architectural milestone, following the stabilization of the micro-graph semantics and the multi-executor interface.