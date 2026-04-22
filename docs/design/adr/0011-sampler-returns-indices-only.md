# ADR-0011: Sampler Must Return Indices Only

## Status
Rejected

## Context
Some parts of the pipeline previously returned partial data instead of indices, coupling sampling with transformation logic.

## Options Considered
### Option 1: Data-returning sampler
- Pros
    - Simple API
- Cons
    - Breaks separation of concerns
    - Cannot be optimized independently
### Option 2: Index-only sampler
- Pros
    - Clean separation of storage vs transformation
    - Enables DAG-based batching
- Cons
    - Requires additional processing step

## Decision
We considered this approach, but rejected it in favor of alternative solutions. All samplers must return only indices (and optional weights), never data.

## Consequences
### Positive
- Fully decoupled sampling layer
- Compatible with execution graph batching

### Negative / Tradeoffs
- Extra indirection step

## Notes
All transformations happen post-sampling via DAG components.