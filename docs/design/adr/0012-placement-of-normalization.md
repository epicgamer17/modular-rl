# ADR-0012: Placement of Normalization (Output Processor, Not DAG Component)

## Status
Proposed

## Context
Normalization can depend on batch statistics, which are not stable or deterministic across DAG nodes. Moving it into the DAG would introduce non-deterministic dependencies.

## Options Considered
### Option 1: Normalize as DAG component
- Pros
    - Unified computation graph
- Cons
    - Requires global stats in DAG
    - Breaks determinism assumptions
### Option 2: Normalize as output processor
- Pros
    - Batch-local, stable computation
    - Keeps DAG deterministic
- Cons
    - Not part of core execution graph

## Decision
We propose adopting this approach because normalization remains an output processor applied after DAG execution.

## Consequences
### Positive
- Stable training behavior
- No global state dependency in DAG

### Negative / Tradeoffs
- Slight separation between normalization and core graph

## Notes
Normalization is statistical post-processing, not structural RL computation.