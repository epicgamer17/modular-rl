# ADR-0023: Derived Execution Graphs from Terminal Targets

## Status
Accepted

## Context
As Reinforcement Learning (RL) architectures evolve, the number of interlocking components (Losses, Target Processors, Diagnostic Loggers, Network Heads) increases significantly. 

Manually specifying the linear execution order of these components ("Hand-Wiring") is a fragile process:
- **Scalability**: A change in one component's requirements can force a manual reordering of the entire pipeline.
- **Hidden Dependencies**: It is easy to accidentally place a consumer before a provider, leading to runtime errors.
- **Waste**: It is difficult to manually determine which diagnostic components can be turned off during an optimized production run without breaking the mathematical chain.

## Options Considered
### Option 1: Manual Sequential Execution (Hand-Wiring)
- **Pros**
    - High transparency; the order is exactly what the code says.
    - No "magic" dependency resolution at startup.
- **Cons**
    - High maintenance burden.
    - Prone to "staleness" bugs where a component uses data from the *previous* step because it was executed before its provider.

### Option 2: Derived Minimal Graph (Chosen)
- **Pros**
    - **Developer Ease**: The user only specifies what they want (e.g., "I want the Policy Loss and the Advantage Histogram").
    - **Optimal Execution**: The system automatically identifies the "Minimal Necessary Set" of components, pruning everything else.
    - **Correctness**: The system uses topological sorting to guarantee that every component is executed only after all its dependencies are met.
- **Cons**
    - startup time increases slightly due to the dependency resolution pass.

## Decision
We will implement the final execution plan will be **derived, not manually wired**.

The user (or higher-level registry) provides:
1. A pool of available **Components**.
2. A list of **Terminal Targets** (requested Blackboard keys or Side Effects).

The Engine then:
1. Performs a **Reverse Dependency Walk** starting from the targets.
2. Identifies all required components based on their `keys_in` and `keys_out` contracts.
3. Performs a **Topological Sort** to establish a safe execution order.
4. Generates the final, executable graph.

## Consequences
### Positive
- **Automatic Pruning**: If a researcher disables a specific loss in the config, the system automatically stops calculating all intermediate tensors that were *only* used for that loss.
- **Refactoring Safety**: Moving a calculation from one component to another doesn't require updating the global ordering; the resolver handles it.
- **Clarity**: The system can output the derived execution graph as a visualization, providing a clear map of how data flows through the system.

### Negative / Tradeoffs
- **Contract Rigidity**: This system fails immediately if any component lacks a clear input/output contract (as mandated by ADR-0016 and ADR-0017).

## Notes
This ADR represents the high-level orchestration layer that utilizes all previous decisions (Contracts, Pure Units, Pure Side Effects) to provide a modern, automated developer experience.