# ADR-0021: Replay Buffer as a Blackboard Boundary

## Status
Accepted

## Context
Traditionally, Replay Buffers are viewed merely as passive storage systems. However, in an architecture built on **Pure Dataflow Units (ADR-0017)** and **Shared Blackboard Contracts (ADR-0016)**, the Replay Buffer plays a more fundamental role. It is the physical realization of the boundary between the "Generation" phase and the "Optimization" phase of Reinforcement Learning.

## Options Considered
### Option 1: Replay as Passive Storage
- **Pros**
    - Simple decoupled implementation.
- **Cons**
    - Requires redundant schema definitions for storage vs. execution.
    - Difficult to verify that sampled data matches what the learner's components expect.

### Option 2: Replay as a Blackboard Boundary (Chosen)
- **Pros**
    - **Architectural Symmetry**: The Actor's "Live Blackboard" and the Learner's "Training Blackboard" use the exact same semantic contracts.
    - **Contract Persistency**: The Replay Buffer schema is not a separate entity; it is simply a version of the Blackboard contract that has been "frozen" and persisted to disk.
    - **End-to-End Validation**: The same validation engine (ADR-0015) can verify the entire pipeline, from the moment an observation is generated in the environment to the moment it is consumed by a loss function, despite the temporal and process gap introduced by the buffer.
- **Cons**
    - Ties the storage format strictly to the execution contract, potentially making replay-side schema migrations more complex.

## Decision
We will implement the Replay Buffer is officially defined as a **Blackboard Boundary**.

1. **Generation Side**: The Actor's execution graph produces a "Generation Blackboard." A specialized side-effect component (`ReplayWriter`) flushes selected keys from this blackboard into the buffer based on a defined contract.
2. **Optimization Side**: The Learner's execution graph is initialized with a "Sampling Blackboard," which is populated by reading a batch from the buffer.
3. **The Schema is the Contract**: There is no separate "buffer schema." Instead, the buffer is defined to store a specific set of **Blackboard Keys with Semantic Types (ADR-0016)**.

## Consequences
### Positive
- **Mental Model Unification**: All data movement in the system, whether it's between local components or between distributed processes, is now understood through the lens of "The Blackboard."
- **Automatic Data Routing**: Components no longer need to know if they are receiving "live" data from an actor or "sampled" data from a buffer; they just read from the Blackboard.
- **Improved Tooling**: Replay inspection tools can leverage the rich metadata (Semantic Axes, Units) attached to the Blackboard contracts.

### Negative / Tradeoffs
- **Tight Coupling**: Forces the Replay Buffer implementation to be "Blackboard-aware," rather than a generic tensor store.

## Notes
This ADR represents the final integration of the storage and execution architectures, fulfilling the vision of **ADR-0007 (Actor-Learner Separation)** within the new Component/Blackboard framework.