# ADR-0004: Tensor-Only Replay Storage

## Status
Accepted

## Context
Current replay system stores Python objects (Sequence, Transition) which introduces:
- serialization overhead
- inconsistent shapes
- inability to use shared memory (torch.multiprocessing)
- ambiguity in contract enforcement

## Options Considered
### Option 1: Python object-based replay
- Pros
  - Flexible
  - Easy to prototype
- Cons
  - Not compatible with multiprocessing shared memory
  - No static shape guarantees
  - Breaks contract system
### Option 2: Tensor-only blackboard storage
- Pros
  - Compatible with torch.shared_memory / multiprocessing
  - Fully vectorizable sampling
  - Enables contract validation
- Cons
  - Requires strict preprocessing pipelines
  - Less flexible at ingestion time

## Decision
We will implement all replay storage must be tensor-only. Python objects are only allowed at environment boundary and must be transformed before storage.

## Consequences
### Positive
- Enables high-throughput distributed training
- Deterministic memory layout
- Compatible with DAG validation

### Negative / Tradeoffs
- More complex ingestion pipeline
- Requires explicit schema design

## Notes
This removes Sequence and Transition from storage path entirely.