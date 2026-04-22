# ADR-0046: Consolidation of Mechanical State (PER & Exploration)

## Status
Proposed

## Context
Features like Prioritized Experience Replay (PER) and Epsilon-Greedy exploration are currently scattered across multiple folders and files (buffers, losses, selectors). This "feature bleed" makes the code harder to reason about and maintain.

## Options Considered

### Option 1: Distributed Logic
- **Pros**: Localizes logic to where it is used (e.g., loss computes priority).
- **Cons**: Spreads state and configuration across multiple modules; leads to inconsistent implementations.

### Option 2: Centralized Mechanical State (Chosen)
- **Pros**: Consolidation of cross-cutting RL features into single, well-defined locations.
- **Cons**: Might centralize too much, requiring careful interface design.

## Decision
Consolidate specific mechanical features into single-owner components:

1. **PER Context**: PER features (priority calculation, metadata management) should exist in one place only, ideally within the Replay Buffer or a dedicated Priority Computer module.
2. **Exploration Context**: All epsilon-greedy features and schedules should be centralized in one place (e.g., within the Action Selector components).
3. **Loss Decoupling**: Loss functions should not implicitly manage priority computation or exploration logic.

## Consequences

### Positive
- **Maintainability**: Changes to exploration or prioritization only require touching one part of the system.
- **Clarity**: Eliminates "feature bleed" across folders/files.

### Negative / Tradeoffs
- **Tight Coupling**: The owner component (e.g., the Buffer) must provide a clean interface if other components need to interact with its state.
