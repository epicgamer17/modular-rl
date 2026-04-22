# ADR-0010: Explicit Temporal Alignment Contract (No Implicit Broadcasting)

## Status
Accepted (Partial Implementation)

## Context
Previous design allowed implicit temporal assumptions (e.g., broadcasting across time, ambiguous T dimension semantics). This led to silent bugs in sequence processing.

## Options Considered
### Option 1: Implicit semantic shapes (e.g., "T means time")
- Pros
    - Flexible
- Cons
    - Ambiguous alignment rules
    - Hidden broadcasting errors
### Option 2: Explicit temporal contract (episode_id + step_id + fixed axes)
- Pros
    - No ambiguity in time dimension
    - Strict validation possible
- Cons
    - Requires stricter schema enforcement

## Decision
We will implement all tensors must explicitly define:
- temporal axis semantics
- step_id alignment
- no implicit broadcasting across time dimension

## Consequences
### Positive
- Eliminates silent shape bugs
- Makes DAG validation reliable

### Negative / Tradeoffs
- More verbose schema definitions

## Notes
Broadcasting is allowed only within a single timestep, not across time.