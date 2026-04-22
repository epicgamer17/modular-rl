# ADR-0045: Universal Pipeline Executors for All Worker Roles

## Status
Accepted

## Context
Currently, actors, learners, and testers often have different execution models, which leads to code duplication and difficulty in swapping components. For example, a "Tester" might use a different logic for action selection than an "Actor", even though both are just performing inference on a network.

## Options Considered

### Option 1: Role-Specific Classes
- **Pros**: Clear separation of responsibilities at the class level.
- **Cons**: High code duplication; difficult to share callbacks or telemetry logic.

### Option 2: Unified Pipeline Executors (Chosen)
- **Pros**: Unifies Actors, Learners, and Testers as specialized pipelines (or sequences of pipelines) that operate on validated computation graphs.
- **Cons**: Requires more abstract design of the pipeline system.

## Decision
All execution agents (Actors, Learners, Testers) will be implemented as pipeline executors that operate on validated computation graphs.

1. **Pipeline Trait**: Every worker (Actor, Learner, Tester) must share a common execution interface.
2. **Unified Semantics**: Clean up callbacks and wrappers to work across all pipeline types.
3. **Role Specialization**: Distinction between roles is handled by the configuration of the graph (e.g., whether it includes an Optimizer node for learning, or a Replay Writer node for acting).
4. **Integration**: Ensure DQN targets work with n-step transitions within the unified pipeline model.

## Consequences

### Positive
- **Code Reuse**: Callbacks, metrics, and environment wrappers can be used identically across all worker types.
- **Flexibility**: Easier to create hybrid workers (e.g., an Actor that performs lightweight on-policy learning).

### Negative / Tradeoffs
- **Complexity**: The pipeline abstraction must be robust enough to handle the varying needs of different workflows.
