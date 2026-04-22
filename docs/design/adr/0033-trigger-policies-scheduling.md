# ADR-0033: Use Trigger Policies for Runtime Scheduling

## Status
Proposed

## Context
Reinforcement Learning (RL) orchestration is rarely a simple linear loop. Systems frequently require complex, interleaved execution patterns based on various conditions:
- **Periodic**: Run a training step every 10 environment steps.
- **Threshold**: Start learning only after the replay buffer has 50,000 transitions (warmup).
- **Event-Based**: Synchronize actor weights only when the learner updates.
- **Adaptive**: Change the learning rate based on reward plateaus.

Hardcoding these patterns into a central "Trainer" or "Runner" loop makes the system rigid and difficult to customize for different algorithms.

## Options Considered

### Option 1: Manual Imperative Loops (Standard)
- **Pros**
    - Straightforward to write and understand for simple cases.
- **Cons**
    - **Hard-Coded Logic**: The orchestration logic (the "When") becomes inextricably mixed with the computation logic (the "What").
    - **Rigidity**: Adding a new periodic event (e.g., "save model every 100 updates") requires modifying the core multi-process runner.

### Option 2: Trigger Policies (Chosen)
- **Pros**
    - **Declarative Scheduling**: The "When" is defined as a pluggable policy (e.g., `PeriodicTrigger(every=10)`).
    - **Reusable Orchestration**: The same trigger logic can be reused for weight syncing, logging, or checkpointing.
    - **Algorithmic Composition**: Complex algorithms can be built by composing simple triggers.
- **Cons**
    - Requires a dedicated **Scheduler Framework** to monitor events and fire triggers.
    - Potential for complex interactions between multiple independent triggers.

## Decision
We propose adopting this approach because the system will utilize **Trigger Policies** for all flexible runtime scheduling.

Rather than a monolithic loop, the high-level orchestration graph will utilize triggers to decide when to invoke specific nodes or subgraphs. 

Standard trigger types include:
1. **`PeriodicTrigger`**: Fires based on a counter (steps, batches, time).
2. **`ThresholdTrigger`**: Fires once a specific metric (e.g., buffer size) is met.
3. **`EventTrigger`**: Fires in response to a signal from another node (e.g., `on_weights_updated`).
4. **`ManualTrigger`**: Fires only when explicitly called by external code.

## Consequences

### Positive
- **Cleaner Trainers**: The main training loop becomes a set of declarative statements rather than a massive block of nested `if/then` statements.
- **Easier Experimentation**: Researchers can change the frequency of weight syncs or the start of the learning phase by modifying a single config value.
- **Consistent distributed behavior**: The same trigger logic works whether it's firing a local function or reaching across process boundaries.

### Negative / Tradeoffs
- **Interaction Complexity**: If multiple triggers depend on each other, it can be difficult to predict the exact interleaving of events without careful visualization.

## Notes
Trigger policies are essential for handling algorithm-specific cadences like weight synchronization, learner frequency vs. actor frequency, and initial warmup periods.