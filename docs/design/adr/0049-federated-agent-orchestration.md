# ADR-0049: Federated Agent Orchestration

## Status
Accepted

## Context
The current entry point for many algorithms is a monolithic "Trainer" class that instantiates all necessary sub-components (Buffer, Network, Optimizer). This tight coupling makes it difficult to swap components for advanced workflows like DAgger, imitation learning, or offline world model training.

## Options Considered

### Option 1: Monolithic Trainers
- **Pros**: Easy to launch a standard algorithm with a single command.
- **Cons**: Rigid; hard to reuse sub-components in different contexts.

### Option 2: Federated Orchestration (Chosen)
- **Pros**: High modularity; allows for swapping any component (e.g., training a world model first, then an agent on top).
- **Cons**: Higher initial configuration complexity for the user.

## Decision
Transition from monolithic trainers to a federated orchestration pattern:

1. **Component Assembly**: Each component (Learner, Buffer, Actor) is instantiated independently.
2. **Generic Drivers**: Use generic "Driver" or "Trainer" scripts that coordinate these pre-instantiated components rather than instantiating them internally.
3. **Flexible Entry Points**: Support diverse workflows like imitation learning (no actor/replay buffer) or DAgger via interchangeable component configurations.

## Consequences

### Positive
- **Flexibility**: Enables complex workflows like pre-training world models separately from agents.
- **Simplicity**: Simplifies examples and configuration by making the relationship between components explicit.

### Negative / Tradeoffs
- **Boilerplate**: Might require slightly more code to "glue" components together in custom scripts.
