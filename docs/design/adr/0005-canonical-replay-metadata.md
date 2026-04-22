# ADR-0005: Canonical Replay Metadata (episode_id, step_id, done)

## Status
Accepted

## Context
Current system uses inconsistent identifiers:

game_ids
implicit step ordering
mixed done, dones, terminated, truncated

This breaks temporal alignment guarantees in the DAG and replay reconstruction.

## Options Considered
### Option 1: Ad-hoc metadata per algorithm
- Pros
    - Flexible per algorithm
- Cons
    - Impossible to unify replay contracts
    - Breaks cross-algorithm training
### Option 2: Canonical metadata schema
- Pros
    - Uniform indexing across all RL methods
    - Enables deterministic replay reconstruction
- Cons
    - Requires migration of existing buffers

## Decision
We will implement all stored transitions must include:
- episode_id
- step_id
- done (single canonical boolean)

## Consequences
### Positive
- Deterministic trajectory reconstruction
- Simplified sampling logic
- Clean DAG temporal reasoning

### Negative / Tradeoffs
- Migration cost across codebase

## Notes
terminated/truncated/dones are derived semantics, not stored facts.