# ADR-0002: Use Component-System Modularity Instead of Inheritance-Centered Algorithm Classes

## Status
Accepted

## Context
The framework must support many RL algorithms and variants:
- DQN
- Rainbow
- PPO
- MuZero
- Search-augmented methods
- Multi-agent methods
- Future hybrids

Traditional RL codebases often organize around algorithm classes:
```python
class PPOTrainer(...)
class DQNTrainer(...)
class MuZeroTrainer(...)
```
Each class owns: rollout logic, replay logic, target generation, loss logic, optimization loop, logging, and scheduling. This creates repeated code and rigid boundaries.

The project instead aims for:
- reusable primitives
- composable algorithms
- explicit dependencies
- swappable modules
- graph execution
- testable units

A component-system style architecture naturally supports these goals.

## Options Considered

### Option 1: Inheritance-Centered Trainer Classes
Each algorithm is a specialized trainer object.
- **Pros**
    - Familiar design
    - Straightforward for small projects
    - Easy local reasoning
- **Cons**
    - Duplicate logic across algorithms
    - Inheritance hierarchies become brittle
    - Hard to mix features across algorithms
    - Tight coupling of concerns
    - Difficult experimentation

### Option 2: Functional Utility Library
Loose helper functions stitched together manually.
- **Pros**
    - Flexible
    - Lightweight
- **Cons**
    - No structural guarantees
    - Hard to discover dependencies
    - Difficult large-scale maintenance

### Option 3: Component-System Architecture (Chosen)
Independent components implement narrow responsibilities:
- loss components
- target builders
- selectors
- replay transforms
- schedulers
- optimizer systems
- environment systems

Algorithms are compositions of components.
- **Pros**
    - Strong modularity
    - High reuse
    - Easy swapping of behavior
    - Natural fit with DAG execution
    - Easier testing
    - Enables feature combinations
- **Cons**
    - More abstraction
    - Requires contracts/interfaces
    - Can feel indirect to new users

## Decision
We will use a component-system architecture where RL functionality is decomposed into reusable components coordinated by the blackboard execution graph. Algorithms are configurations of components, not subclassed trainer classes.

## Consequences

### Positive
- PPO, DQN, MuZero can share primitives.
- Easier experimentation with hybrids.
- Losses, targets, and selectors become reusable units.
- Cleaner testing boundaries.
- Natural integration with execution graph.

### Negative / Tradeoffs
- More files / smaller units.
- Requires disciplined contracts.
- Debugging may require graph inspection.

## Notes
This is ECS-inspired, but not classic ECS. The system does not primarily model millions of archetyped entities like a game engine.

Instead:
- **Components** = reusable RL behaviors
- **Systems** = executable graph nodes
- **Blackboard** = shared typed state

So the architecture is best described as: **Blackboard DAG + ECS-style components**.
