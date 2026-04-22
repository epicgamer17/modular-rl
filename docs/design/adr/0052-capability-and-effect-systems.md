# ADR-0052: Capability and Effect Systems for Algorithmic Integrity

## Status
Proposed

## Context
RL algorithms have many subtle "hidden" requirements (e.g., PPO needs fresh data, DQN needs discrete actions). Currently, these are enforced manually in code or through training failures. There is no declarative way for a component to state "I cannot run on continuous actions" or "I require data less than 50 steps old."

## Options Considered

### Option 1: Runtime Asserts
- **Pros**: Easy to implement.
- **Cons**: Errors only appear during training; no compile-time safety.

### Option 2: Capability-Based Validation (Chosen)
- **Pros**: Enables the graph compiler to reject mathematically or algorithmically invalid configurations before launch.
- **Cons**: Requires formalizing "Effect" and "Capability" metadata.

## Decision
We propose adopting a system for declaring and validating algorithmic capabilities and side-effects.

1. **Mathematical Capabilities**: Components (like DQN) will explicitly declare their supported action/state types (e.g., `DiscreteOnly`). The Graph Compiler will verify this against the environment contract.
2. **Data Freshness / Effects**: Introduce "Freshness" contracts. On-policy components (like PPO) will declare a dependency on "Fresh" data, and the compiler will verify that the replay source being used isn't configured for large off-policy buffers.
3. **Privileged Data Isolation**: Explicitly tag specific keys as `Privileged` (e.g., opponent state, ground-truth reward labels). The system will enforce that evaluators, production actors, or non-privileged components cannot subscribe to these keys.
4. **Execution Separation**: Ensure the same semantic graph can be mapped to different backends (Local, Ray, C-compiled) because capabilities are expressed independently of the runtime.

## Consequences

### Positive
- **Correctness**: Prevents subtle algorithmic bugs (like accidentally training PPO on stale data).
- **Safety**: Robustly prevents "cheating" in evaluations by blocking access to privileged data.
- **Portability**: Clearly separates what an algorithm *is* from how it *runs*.

### Negative / Tradeoffs
- **Overhead**: Requires developers to specify more metadata for components.
