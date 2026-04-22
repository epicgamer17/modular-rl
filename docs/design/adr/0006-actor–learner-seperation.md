# ADR-0006: Actor–Learner Separation via Shared Replay Contract

## Status
Accepted

## Context
Actor-side trajectory generation and learner-side training currently share implicit assumptions about data format, leading to tight coupling and hidden bugs.

## Options Considered
### Option 1: Shared Python structures (Sequence/Transition)
- Pros
    - Easy sharing between actor and learner
- Cons
    - Not scalable
    - No strict validation boundary
### Option 2: Shared replay contract (tensor schema)
- Pros
    - Clear boundary between actor and learner
    - Enables distributed systems
- Cons
    - Requires strict schema enforcement

## Decision
We will implement actor emits structured episode streams which are immediately converted into tensor-based replay contract format before storage.

## Consequences
### Positive
- Clear system boundary
- Supports distributed actors
- Enables replay determinism

### Negative / Tradeoffs
- Requires strict sche  ma versioning

## Notes
Replay becomes a contract bridge, not a data container.