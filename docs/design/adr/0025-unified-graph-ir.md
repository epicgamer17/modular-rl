# ADR-0025: Unified Graph IR Across Macro and Micro Layers

## Status
Proposed

## Context
The system originally utilized different semantic models for high-level orchestration (Macro layer - e.g., connecting an actor to a learner) and low-level computation (Micro layer - e.g., the DAG of components within the learner). This split created a conceptual gap, duplicated architectural patterns, and made it difficult to optimize across these boundaries.

## Options Considered

### Option 1: Separate Macro Graph and Micro Blackboard Graph
- **Pros**
    - Faster initial migration from existing distributed code.
    - Minimal immediate refactoring of the high-level orchestration logic.
- **Cons**
    - Dual semantics increase the learning curve for contributors.
    - Harder to perform cross-boundary optimizations (e.g., pruning nodes across the distributed boundary).
    - Accumulates long-term technical debt by maintaining two execution models.

### Option 2: Single Recursive Graph IR (Chosen)
- **Pros**
    - **Uniform Semantics**: The same abstraction applies whether you are wiring together distributed actors or individual loss components.
    - **Nested Graphs**: Naturally supports hierarchical decomposition of complex RL systems.
    - **Reusable Tooling**: Graph validation, visualization, and compilation tools work universally across all layers.
- **Cons**
    - Larger initial refactoring effort.
    - Requires a robust, generalized node model that accommodates both local and remote execution.

## Decision
We propose adopting this approach because the system will utilize a single **Unified Graph Intermediate Representation (IR)** for all levels of abstraction. 

Execution plans will be represented as graphs where **Composite Nodes** can contain subgraphs of the exact same graph type. This recursive structure allows the same "Graph Engine" to handle everything from micro-level component execution to macro-level distributed workflow orchestration.

## Consequences

### Positive
- **Conceptual Simplicity**: Developers only need to understand one execution model.
- **Evolution**: Infrastructure improvements (like better scheduling or pruning) benefit the entire system simultaneously.
- **Transparency**: High-level system behavior can be inspected and verified using the same contract-driven tools as low-level components.

### Negative / Tradeoffs
- **Initial Refactor**: Requires a significant redesign of the high-level orchestration layer.
- **Abstraction Leakage**: Care must be taken to ensure the IR remains generic enough to handle different execution modes (local threads vs. Ray actors) without becoming overly complex.

## Notes
This was a major turning point in the architecture discussion, aligning the distributed system's design with the ECS-style component model.