# ADR-0016: Components as Pure Dataflow Units

## Status
Accepted

## Context
In a Directed Acyclic Graph (DAG) based execution model, the system must be able to reason about the inputs and outputs of every node. If components perform hidden mutations or have unpredictable side effects, the system loses the ability to:
- **Prune Nodes**: Safely remove components whose outputs are not needed for a specific target.
- **Reorder Nodes**: Optimize execution for performance or memory locality.
- **Parallelize**: Safely execute independent branches of the graph.
- **Debug**: Identifying where a value was changed becomes impossible if changes happen outside the primary data channel (the Blackboard).

## Options Considered
### Option 1: Stateful/Opaque Components
- **Pros**
    - Familiar for traditional object-oriented programming.
    - Easier to implement components that need to maintain internal counters or buffers.
- **Cons**
    - Makes graph reasoning and optimization impossible.
    - Hidden bugs are common due to shared mutable state.

### Option 2: Pure Dataflow Units
- **Pros**
    - **Determinism**: For a given set of inputs on the Blackboard, a component always produces the same outputs.
    - **Optimization-Ready**: The system can safely prune or reorder components based on their declared dependencies.
    - **Testability**: Components can be unit-tested by simply mocking the Blackboard inputs and checking outputs.
- **Cons**
    - Requires explicit handling for "true" side effects (logging, optimization).

## Decision
We will implement components are strictly defined as **Pure Dataflow Units** by default.

A standard component must follow these rules:
1. **Read-Only Context**: It reads data only from its declared `keys_in`.
2. **Deterministic Computation**: It performs math or logic on the inputs.
3. **Write-Only Output**: It writes results only to its declared `keys_out`.
4. **No Internal State**: It does not maintain mutable internal state that affects its logic across steps (unless it is a fundamental architectural state like an RNN cell, which should still be managed via the Blackboard if possible).

### Explicit Side Effects
Components that *must* perform side effects are treated as special cases and must be explicitly marked. Examples include:
- **Optimization**: `OptimizerStepComponent` (mutates model weights).
- **Storage**: `ReplayWriterComponent` (writes to external buffer/disk).
- **Diagnostics**: `LoggingComponent` (sends data to Weights & Biases/TensorBoard).
- **Check-pointing**: `ModelSaverComponent`.

## Consequences
### Positive
- **Safe Side-Effect Pruning**: The validation engine (ADR-0015) can safely remove any node that doesn't lead to a "Side-Effect Task" or a requested "Terminal Key."
- **Clarity**: The entire state of the system at any point in the graph is represented by the Blackboard.
- **Concurrency**: Branches with no overlapping keys/side-effects can be executed in parallel with zero risk of race conditions.

### Negative / Tradeoffs
- **Registration overhead**: Side-effect operations that were previously "hidden" inside loss functions must now be broken out into explicit components.

## Notes
This decision is the foundation for the **Blind Learner** and **Blind Actor** philosophy, ensuring that logic units are interchangeable and predictable regardless of where they are executed.