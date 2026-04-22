# ADR-0050: Global and Temporal Blackboard Storage

## Status
Proposed

## Context
Currently, the Blackboard focuses on per-sample or per-batch data during a single execution step. However, RL algorithms often require state that persists across multiple training steps (e.g., schedules for temperature) or history that spans multiple epochs (e.g., target network weights, statistics for evaluators).

## Options Considered

### Option 1: Monolithic State Object
- **Pros**: Simple to access everything from one place.
- **Cons**: Poor separation of concerns; difficult to manage distributed updates.

### Option 2: Multi-Domain Blackboard (Chosen)
- **Pros**: Explicitly separates transient data from persistent global state and temporal history.
- **Cons**: Requires more complex key resolution logic.

## Decision
We propose adopting a multi-domain blackboard architecture to formalize "shared storage" and temporal tracking.

1. **Global Domain**: Dedicated to cross-episode state such as training temperature, shared network weights, and global hyperparameter counters. This is the source of truth for "Shared Storage" in MuZero-style pseudocode.
2. **Temporal Domain**: Dedicated to historical facts, tracking per-episode statistics (scores, lengths), and storing non-online model weights for target-network access or evaluator-matrix calculations.
3. **Registry Integration**: Both domains will be accessible via standard Blackboard keys (e.g., `global.temperature`, `temporal.history.score`) with explicit read/write permissions.

## Consequences

### Positive
- **Algorithm Parity**: Directly maps to MuZero/AlphaZero "shared storage" concepts.
- **Observability**: Simplifies tracking of long-term metrics and stats without polluting the training data path.
- **Weight Management**: Formalizes the management of target and evaluator weights.

### Negative / Tradeoffs
- **Backend Complexity**: The Blackboard engine must now handle multiple storage backends with different persistence traits.
