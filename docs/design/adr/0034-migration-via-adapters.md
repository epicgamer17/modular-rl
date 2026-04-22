# ADR-0034: Migrate Existing Engine via Adapters, Not Rewrite

## Status
Proposed

## Context
The system has undergone a major architectural shift toward a **Unified Graph IR (ADR-0026)** and **Multiple Executors (ADR-0031)**. However, the existing (legacy) execution engine already contained significant logic for:
- Efficient workspace/blackboard management.
- Local dependency resolution.
- Performance optimizations for single-device training.

Attempting a complete "clean slate" rewrite of all internal scheduling logic would be high-risk, potentially introducing subtle regressions in core training stability and performance that would be difficult to track down.

## Options Considered

### Option 1: Rewrite Everything Immediately (Clean Slate)
- **Pros**
    - Eliminates all legacy technical debt in one go.
    - Resulting code is purely aligned with the new Graph IR.
- **Cons**
    - **High Risk**: Regression in training stability or performance could stall the project for weeks.
    - **Low Velocity**: Progress on new features would stop until the rewrite was 100% complete.

### Option 2: Use Adapters for Progressive Migration (Chosen)
- **Pros**
    - **Lower Risk**: Existing, proven code handles the heavy lifting during the transition.
    - **Faster Progress**: New architectural features (like distributed actors) can be built on top of the new interface immediately.
    - **Preserves Functionality**: Ensures that existing algorithms (PPO, Muzero) continue to work correctly throughout the refactoring.
- **Cons**
    - Introduces temporary code complexity in the form of "Adapter" layers.
    - Requires maintaining two parallel execution paths during the transitional period.

## Decision
We propose adopting this approach because the system will **migrate the existing engine via Adapters** rather than a full rewrite.

Legacy execution logic will be wrapped in an interface that satisfies the new **Graph IR** requirements. Specifically:
1. A `WorkspaceExecutorAdapter` (and similar classes) will take a new Graph definition, translate it into the legacy engine's internal format, and execute it using the proven workspace-backed logic.
2. Internals of the legacy engine will then be progressively refactored to align with the new structural decisions (e.g., separating Transforms from Nodes) one piece at a time.
3. The legacy "outer" APIs will be deprecated and eventually removed once the new Graph Engine reaches feature parity.

## Consequences

### Positive
- **Stable Foundation**: Exploits the reliability of the existing codebase while moving toward the new architecture.
- **Fast Iteration**: Researchers can begin using the new DAG-based configuration immediately.
- **Feature Continuity**: Important features like automatic mixed precision and hardware dispatching remain functional during the transition.

### Negative / Tradeoffs
- **Transitional Code Paths**: The codebase will contain several "shim" and "adapter" layers that must be monitored and eventually cleaned up.

## Notes
The `WorkspaceExecutorAdapter` is the primary realization of this principle, acting as the bridge between the old "Linear Workspace" model and the new "Unified Graph" model.