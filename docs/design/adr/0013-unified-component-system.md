# ADR-0013: Unified Component System for Actors and Learners

## Status
Accepted

## Context
Many Reinforcement Learning (RL) frameworks split Actor (data collection) and Learner (optimization) into separate, redundant codebases. This creates significant duplication of logic for operations such as:
- Loss calculations (when used for diagnostic logging in actors).
- Return estimation (GAE, N-step).
- Network unrolling and state management.
- Action selection and masking.

Maintaining two parallel implementations of the same mathematical logic is error-prone and increases technical debt.

## Options Considered
### Option 1: Separate Actor and Learner logic (Status Quo)
- **Pros**
    - High isolation; acting logic cannot accidentally corrupt learning logic.
    - Optimized for specific hardware (CPU for actors, GPU for learners).
- **Cons**
    - High code duplication.
    - Inconsistent behavior between training and inference (e.g., different return calculations).

### Option 2: Unified Component System
- **Pros**
    - Single source of truth for all mathematical and RL logic.
    - Components (Losses, Networks, Selectors) are pluggable and reusable.
    - Reduces the surface area for bugs.
- **Cons**
    - Requires a robust execution plan system to handle different keys/configs for actors vs learners.
    - Potentially higher initial complexity in component design.

## Decision
We will implement actors and Learners will both utilize the same **Blackboard** and **Component** architecture. 

While they remain separate execution processes (as per ADR-0007), they will share the same underlying logic blocks. They will differ only in their:
1. **Target Keys**: The specific blackboard keys they read from and write to.
2. **Execution Plans**: The sequence and set of components invoked during a step.

## Consequences
### Positive
- **Code Reuse**: A single `ValueLoss` or `GAEProcessor` can be used by both the learner (for backprop) and the actor (for advantage normalization or logging).
- **Consistency**: Guaranteed mathematical parity between acting and learning logic.
- **Developer Velocity**: New features only need to be implemented as a `Component` once to be available across the entire system.

### Negative / Tradeoffs
- **Complexity**: Components must be designed with "blindness" to their environment, relying strictly on the Blackboard context.
- **Configuration Overhead**: Careful management of `keys_in` and `keys_out` is required to ensure the correct data flows through the DAG in both actor and learner contexts.

## Notes
This decision reinforces the "Blind Learner" and "Blind Actor" philosophy defined in the project structure rules, ensuring that both units are just orchestrators of the same shared logic components.