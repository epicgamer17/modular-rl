# ADR-0022: Explicit Target Calculation as DAG Components

## Status
Accepted

## Context
Reinforcement Learning reliance on "targets" (e.g., N-step returns, Generalized Advantage Estimation (GAE), Bootstrap Values) is fundamental. However, these calculations are frequently treated as "invisible utility code" or hidden preprocessing steps inside Replay Buffers or Loss Functions.

This "hidden preprocessing" approach leads to several problems:
- **Ambiguity**: It is unclear where a target was calculated or which version of an algorithm was used.
- **Fragility**: Swapping a target calculation (e.g., from N-step to V-trace) often requires invasive changes to core training loops.
- **Inspection Barrier**: Verifying that advantages are correctly normalized or that GAE lambda is being applied properly is difficult when the math is buried.

## Options Considered
### Option 1: Utility Functions or Replay Processors (Standard)
- **Pros**
    - Performance; can be highly optimized as specialized C++ or NumPy routines.
- **Cons**
    - Hard to validate within the context of the execution graph.
    - Hidden from the DAG pruner and visualization tools.

### Option 2: Explicit DAG Components (Chosen)
- **Pros**
    - **Uniformity**: Target calculations are treated exactly like Neural Network layers or Loss functions.
    - **Contract-Driven**: Input dependencies (e.g., `Values`, `Rewards`) and output contracts (e.g., `GAEAdvantages`) are explicitly declared and verified (ADR-0015).
    - **Pluggability**: Switching from N-step to GAE is simply a matter of swapping one node for another in the execution plan.
    - **Observability**: Targets on the Blackboard can be easily intercepted by `LoggingComponents` for diagnostics.
- **Cons**
    - Slight overhead in managing more nodes in the graph.

## Decision
We will implement all target-generation math must be implemented as explicit **DAG Components**.

Examples of these components include:
- `GAEComponent`: Takes `Rewards`, `Values`, and `Dones`; outputs `Advantages`.
- `NStepReturnComponent`: Calculates N-step discounted returns.
- `ValueBootstrapComponent`: Handles the terminal state bootstrapping logic.

Any "preprocessing" that was previously hidden in the Replay Buffer sampling path must now be moved into the Learner's execution graph as a set of formal components.

## Consequences
### Positive
- **Architectural Clarity**: The mathematical pipeline—from raw data to loss—is fully represented in the execution graph.
- **Simplified Learners**: Learners no longer need to contain complex math for return estimation; they simply request the appropriate target keys from the Blackboard.
- **Testability**: Target components are pure dataflow units (ADR-0017) that can be unit-tested with deterministic tensor inputs.

### Negative / Tradeoffs
- **Graph verbosity**: The execution plans will be longer as they now include the explicit data-munging steps.

## Notes
This decision reinforces the **Unified Component System (ADR-0014)**, ensuring that even the "boring" math of return estimation follows the same rigorous standards as the "exciting" math of the neural networks.