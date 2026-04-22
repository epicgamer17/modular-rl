# ADR-0009: Removal of Sequence/Transition Objects from Replay Path

## Status
Accepted (Partial Implementation)

## Context
Sequence and Transition objects:
- introduce Python object overhead
- cannot be shared across processes efficiently
- violate tensor-only contract

## Options Considered
### Option 1: Keep objects as intermediate abstraction
- Pros
    - Convenient debugging
- Cons
    - Leakage into storage pipeline

### Option 2: Remove entirely from replay path
- Pros
    - Clean tensor pipeline
    - Full multiprocessing compatibility
- Cons
    - Less human-readable intermediate state

## Decision
We will implement remove Sequence/Transition from replay storage and sampling paths. They may exist only at environment debugging layer.

## Consequences
### Positive
- Clean separation of concerns
- Better performance

### Negative / Tradeoffs
- Harder debugging without tooling

## Notes
Debug views must be constructed from tensors, not stored objects.