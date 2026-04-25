# Step 14.1 Verification Report: Schedule IR

This report verifies the introduction of the declarative `Schedule IR` system, which replaces ad-hoc loops with a structured `SchedulePlan`.

## 1. Feature Mapping

| Component | Responsibility | Semantic Role |
| :--- | :--- | :--- |
| **SchedulePlan** | Declarative configuration of frequencies and strategies | Orchestration Logic |
| **ScheduleRunner** | Operationalizes the plan by driving runtimes | Execution Engine |
| **Schedule IR Node** | Represents the schedule within the computation graph | Declarative Intent |

## 2. Alignment with Semantic Integrity

- **Declarative Scheduling**: The execution pattern is no longer buried in Python `for` loops. It is defined as a `SchedulePlan` that can be inspected, optimized, and serialized.
- **Explicit Frequencies**: `actor_frequency` and `learner_frequency` clearly define the ratio of interaction to training, preventing implicit coupling.
- **Synchronization Points**: The introduction of `sync_points` allows for formalizing where parameter updates or metric logging should occur, ensuring causal consistency across parallel workers.

## 3. Orchestration Logic Verified (Test 14.1)

The implementation was verified using a mock-based execution test:
- **Frequency Ratio**: Verified that setting `actor_frequency=5` and `learner_frequency=2` correctly results in 5 actor steps and 2 learner steps per iteration.
- **Step Counting**: Proved that the `ScheduleRunner` accurately tracks total steps and terminates precisely at the requested limit.
- **Serialization**: Verified that `SchedulePlan` correctly serializes to a dictionary for logging or RPC transport.

## 4. Implementation Details
- [x] **IR Integration**: Added `NODE_TYPE_SCHEDULE` to the core graph schema.
- [x] **Stateful Orchestration**: Implemented `ScheduleRunner` in `runtime/scheduler.py` to coordinate `ActorRuntime` and `LearnerRuntime`.
- [x] **Export Cleanliness**: Updated `runtime/__init__.py` to export the new scheduling primitives while removing deprecated ad-hoc loop constructs.

## 5. Legacy Porting & Parallelism Strategy

The transition from legacy ad-hoc scheduling to the unified IR system involved a complete port of existing functionality into the `ScheduleRunner` logic.

### Consolidation of Primitives
- **EveryN & Loop**: These manual triggers were absorbed into the `actor_frequency` and `learner_frequency` parameters of the `SchedulePlan`. Instead of nested loop logic, the executor now manages the ratio of interaction to training as a first-class citizen.
- **ParallelActorPool**: The thread-management logic was moved directly into `ScheduleRunner._execute_actors`. 

### How Parallel Actors are Handled Now
With the removal of `ParallelActorPool`, parallelism is now a **declarative strategy**:
1.  **Multiple Runtimes**: Users provide a list of `ActorRuntime` instances to the `ScheduleRunner`.
2.  **Strategy Selection**: By setting `batching_strategy="parallel"`, the executor spawns a multi-threaded batch execution.
3.  **Thread Safety**: Each thread operates on its own `ActorRuntime` (and its associated `ExecutionContext`), while sharing access to the `ParameterStore` and `ReplayBuffer` through existing thread-safe locks.

### Verification of Parallelism
- **test_schedule_plan_parallel_strategy**: This test confirms that when the `parallel` strategy is active, the executor correctly utilizes threading to drive multiple actors concurrently.
- **Causal Consistency**: Verified that parallel execution preserves the `ActorSnapshot` integrity, ensuring that each thread uses its intended policy version regardless of concurrent updates from the learner.

> [!IMPORTANT]
> By moving to a Schedule IR, we enable future optimizations like "auto-scheduling" where the system adjusts frequencies based on hardware utilization or convergence rates.
