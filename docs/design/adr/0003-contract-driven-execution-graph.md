# ADR-0003: Contract-Driven Execution Graph for RL Pipelines

## Status
Accepted

## Context
The system requires a unified way to compose RL computation (losses, targets, sampling transforms) without hardcoding execution order or implicitly coupling components. Existing pipelines mix procedural replay logic with learning logic, making dependency tracking and minimal execution difficult.

Constraints:
- Must support multiple RL families (PPO, DQN, MuZero)
- Must enable minimal recomputation based on requested outputs
- Must remain compatible with distributed sampling/training

## Options Considered
### Option 1: Procedural pipeline (current-style)
- Pros
  - Simple to implement
  - Easy to debug step-by-step
- Cons
  - No dependency resolution
  - Over-computation (compute unused targets)
  - Hard coupling between components

### Option 2: Execution graph (DAG of components)
- Pros
  - Explicit dependencies between losses/targets
  - Enables pruning of unused computation
  - Naturally composable across algorithms
- Cons
  - Higher initial complexity
  - Requires strict contract definitions

## Decision
We will adopt a DAG-based execution graph where all components declare inputs/outputs via contracts and are executed only if required by target keys.

## Consequences
### Positive
- Minimal computation per training step
- Clear dependency tracking
- Composable RL algorithms
### Negative / Tradeoffs
- More complex debugging model
- Requires strict contract discipline

## Notes
This is the foundation for “blackboard_engine + execution_graph” architecture.