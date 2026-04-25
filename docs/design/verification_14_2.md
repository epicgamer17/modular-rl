# Step 14.2 Verification Report: Compiler-Driven Scheduling

This report verifies the relocation of scheduling decisions from the runtime into the compiler output, ensuring the runtime remains a pure, passive executor.

## 1. Feature Mapping

| Component | Responsibility | Semantic Role |
| :--- | :--- | :--- |
| **Schedule Compiler** | Derives execution strategy from Graph topology and tags | Decision Maker |
| **SchedulePlan** | Static artifact produced by the compiler | Compiled Plan |
| **ScheduleRunner** | Operationalizes the provided plan without modification | Pure Execution |

## 2. Alignment with Semantic Integrity

- **Passive Runtime**: The runtime no longer "guesses" how often to run actors vs learners. It receives a `SchedulePlan` (the "compiled binary" of orchestration) and follows it exactly.
- **Deterministic Derivation**: Given a graph and a set of user hints, the compiler produces a deterministic `SchedulePlan`. This ensures that two identical graphs with identical hints will always execute identically.
- **Graph Immutability**: Scheduling changes (e.g., increasing `prefetch_depth`) do not require mutations to the logical IR graph, preserving the separation between "What to compute" and "How to schedule it".

## 3. Deterministic Execution Verified (Test 14.2)

The implementation was verified against the requirement for deterministic changes:
- **Hint Propagation**: Verified that `user_hints` provided to the `compile_schedule` pass are correctly translated into the final `SchedulePlan`.
- **Automatic Injections**: Proved that the compiler automatically injects safety constraints (e.g., `step` sync for on-policy graphs) even when not explicitly requested by the user.
- **IR Stability**: Confirmed that the `Graph` object remains physically unchanged after a compilation pass, even when the resulting `SchedulePlan` varies significantly.

## 4. Implementation Details
- [x] **New Compiler Module**: Created `compiler/scheduler.py` as the dedicated pass for schedule generation.
- [x] **Heuristic Logic**: Implemented automatic detection for `parallel` strategy, `prefetch_depth`, and `sync_points` based on graph topology.
- [x] **Registry Cleanliness**: The `SchedulePlan` remains the bridge between the compiler's decisions and the runtime's execution.

> [!TIP]
> This separation allows us to "A/B test" different scheduling strategies (e.g., serial vs parallel) by generating multiple plans for the same graph and comparing performance without risking logic regressions.
## Modernization of Examples

All core examples have been updated to utilize the new declarative systems:
- **PPO**: Demonstrates `ActorRuntime` with automatic snapshot binding and `ScheduleRunner` with on-policy constraints.
- **DQN**: Demonstrates `ReplayQuery` nodes and compiler-driven prefetching.
- **DAgger**: Demonstrates custom `recording_fn` for dataset aggregation and multi-actor coordination.
- **NFSP**: Demonstrates dual-buffer management and complex orchestration using split runtimes.

These examples serve as the primary reference for building high-performance RL agents using the Semantic Kernel.
