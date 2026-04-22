# ADR-0048: Standardized Hyperparameter Scheduling

## Status
Accepted

## Context
Learning rates, clip ranges, and other hyperparameters have inconsistent naming and initialization schemes for their schedules across different algorithms. This makes it difficult to apply scheduling to new parameters or reuse schedule logic.

## Options Considered

### Option 1: Ad-hoc Scheduling
- **Pros**: Localized control over parameter decay.
- **Cons**: Inconsistent behavior; duplication of "LinearAnneal" classes.

### Option 2: Unified Schedule Interface (Chosen)
- **Pros**: Standardized scheme for making any scalar hyperparameter follow a schedule.
- **Cons**: Requires refactoring existing parameter management.

## Decision
All hyperparameter schedules will follow a unified API and naming scheme:

1. **Universal Schedule API**: Any scalar hyperparameter can be replaced with a `Schedule` object (e.g., `LinearSchedule`, `CosineSchedule`).
2. **Initialization Consistency**: Standardize how schedules are initialized from configuration to ensure all parameters (epsilon, LR, clip range) use identical semantics.
3. **Registry Integration**: Schedules should be easily accessible from a central registry.

## Consequences

### Positive
- **Consistency**: All decaying parameters behave predictably.
- **Ease of Use**: Making a new parameter follow a schedule becomes trivial.

### Negative / Tradeoffs
- **Implementation Effort**: Requires migrating all existing schedules to the new unified scheme.
