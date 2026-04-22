# ADR-0031: Preserve Contract System Across All Layers

## Status
Proposed (Partial Implementation)

## Context
As architectural layers were introduced—from pure mathematical **Transforms (ADR-0029)** to **Unified Graph IR (ADR-0026)** and **Composite Nodes (ADR-0028)**—there was a risk that validation logic would fragment. 

A "Contract" that validates a local loss calculation (shape, dtype, semantic meaning) should have the same expressive power and language as a contract that validates the connection between a distributed Actor and a Replay Buffer. Having multiple validation systems would lead to inconsistent guarantees and duplicated orchestration logic.

## Options Considered

### Option 1: Separate Contract Systems for Different Layers
- **Pros**
    - Allows for faster, specialized implementations (e.g., a lightweight one for micro-layers and a robust one for RPC-based macro-layers).
- **Cons**
    - **Logic Duplication**: Identical validation logic (e.g., shape checking) must be implemented twice.
    - **Conceptual Fragmentation**: Users must learn different ways to declare dependencies depending on where they are in the system hierarchy.

### Option 2: Unified Contract System (Chosen)
- **Pros**
    - **Consistent Validation**: A `PolicyLogits` tensor is validated identically whether it is passed between local layers or across a distributed workflow.
    - **Better Tooling**: Visualization and debugging tools can interpret any part of the system using a single schema definition.
    - **Seamless Composition**: A micro-graph can be promoted to a macro-layer node (see ADR-0028) without rewriting its interface definitions.
- **Cons**
    - Requires a more generalized and robust schema design that can handle diverse execution modes and serialization requirements.

## Decision
We propose adopting this approach because the system will use a **Single Contract Language** for all levels of abstraction.

This unified language will be used to define the interfaces for:
1. **Transforms**: The inputs and outputs of local math functions.
2. **Nodes**: The execution boundaries within a graph.
3. **Ports**: The boundary mappings for Composite Nodes (nested graphs).
4. **Subgraphs**: The aggregate requirements of an entire execution plan.

Every contract in the system will include metadata for:
- **DType**: (e.g., `torch.float32`).
- **Shape**: (including **Semantic Axes**, see ADR-0020).
- **Semantic Meaning**: (e.g., `"The raw policy outputs before softmax"`).
- **Broadcasting Policy**: (as per ADR-0021).

## Consequences

### Positive
- **Strong System-Wide Guarantees**: A valid graph at the macro level is guaranteed to be valid at the micro level, eliminating a huge class of "plumbing" bugs.
- **Simpler Mental Model**: Developers learn one system for defining "what piece of data this node needs."
- **Automatic Documentation**: The system can generate a comprehensive data dictionary for the entire training codebase automatically.

### Negative / Tradeoffs
- **Schema Overhead**: Designing a unified contract that handles both local tensor references and remote serialization contracts is technically challenging and requires careful upfront design.

## Notes
The unified contract system is the binding force that makes the **Recursive Composite Node (ADR-0028)** and **Unified Graph IR (ADR-0026)** functional and safe.