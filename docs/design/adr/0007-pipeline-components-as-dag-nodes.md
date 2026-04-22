# ADR-0007: Pipeline Components as DAG Nodes (ECS-style RL Systems)

## Status
Accepted

## Context
Current processors mix responsibilities (n-step, masking, compression, normalization) without clear execution semantics. Some belong to replay, others to learning.

## Options Considered
### Option 1: Processor-based pipeline (current)
- Pros
    - Simple chaining
- Cons
    - No dependency resolution
    - Hard to optimize execution
### Option 2: DAG components (ECS-style systems)
- Pros
    - Each transform is a node in execution graph
    - Dependencies are explicit
    - Enables recomputation minimization
- Cons
    - Requires redesign of pipeline architecture

## Decision
We will implement move target-heavy transforms (n-step, GAE, unroll, masking) into components/ as DAG nodes. Data folder only handles storage + sampling.

## Consequences
### Positive
- Unified computation model
- Reusable RL primitives
- Better optimization opportunities

### Negative / Tradeoffs
- Larger concep tual overhead

## Notes
This is the “bridge layer” between replay and learning.