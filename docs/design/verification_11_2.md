# Step 11.2 Verification Report: Runtime Split (Actor vs Learner)

This report verifies the structural decoupling of the RL Controller into independent `ActorRuntime` and `LearnerRuntime` components, and the subsequent removal of the legacy `RolloutController`, enabling asynchronous and distributed RL patterns.

## 1. Feature Mapping

| Component | Responsibility | Semantic Role |
| :--- | :--- | :--- |
| **ActorRuntime** | Env Interaction, Trace Generation | Online Execution (Rollout) |
| **LearnerRuntime** | Sampling, Optimization, Updates | Offline Execution (Training) |
| **ParameterStore** | Versioning, Synchronization | Causal Consistency Layer |
| **ReplayBuffer** | Storage, Thread-Safe Access | Data Decoupling Layer |

## 2. Alignment with Semantic Integrity (Section 1.1)

The split runtime architecture solves the following design goals:
- **Decoupled Loops**: Actors can run at high frequency (limited by CPU/Env) while Learners run at their own pace (limited by GPU/Throughput).
- **Explicit Staleness Control**: By separating the runtimes, the system can now explicitly track and validate "policy staleness" (the delta between data generation version and current training version).

## 3. Asynchronous Correctness (Test 11.2)

Test 11.2 verified the system's behavior under an asynchronous staleness model:
- **Thread Safety**: `ReplayBuffer` was upgraded with internal locking to prevent race conditions during concurrent Actor production and Learner consumption.
- **Staleness Detection**: Verified that the `LearnerRuntime` correctly identifies and rejects stale data for On-Policy algorithms (like PPO) when the global `ParameterStore` version has advanced beyond the data's snapshot version.
- **Continuous Production**: Demonstrated a multi-threaded setup where an actor produces data in the background while the learner samples and validates it in real-time.

## 4. Verification of Implementation Details
- [x] **Full Migration**: `RolloutController` was completely removed, and the entire codebase (examples, tests, scheduler) was migrated to natively use `ActorRuntime` and `LearnerRuntime`.
- [x] **Atomic Mutations**: `ReplayBuffer.add` was made atomic to prevent learners from sampling partially-initialized transition records.
- [x] **Context Propagation**: Both runtimes correctly propagate the `ExecutionContext`, ensuring traces carry the exact policy versions required for verification.

> [!TIP]
> This split is the prerequisite for **Distributed RL**. In the next phase, `ActorRuntime` and `LearnerRuntime` can be moved to separate processes or network nodes, with the `ParameterStore` acting as the synchronization backbone.
