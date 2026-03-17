# 🧠 Learners: The Universal GPU Engine

This module contains the core optimization engine for the reinforcement learning pipeline. It is designed with a strict **Event-Driven, Data-Oriented Architecture**. 

There are no algorithm-specific subclasses (e.g., `PPOLearner` or `MuZeroLearner` do not exist). Instead, a single `UniversalLearner` handles all optimization by acting as a pure mathematical pipeline. Algorithm-specific behavior is injected via strict interfaces: **Iterators**, **Target Builders**, **Loss Pipelines**, and **Callbacks**.

## 🏗️ Core Contracts & Philosophy

### 1. `UniversalLearner` (The Engine)
The Learner is a blind, stateless GPU engine. It knows nothing about Replay Buffers, environments, or telemetry dashboards.
* **Input Contract:** It accepts an `Iterable` that yields standard Python dictionaries (`Dict[str, Tensor]`).
* **Execution Contract:** It processes data through a rigid sequence: `Forward Pass -> Target Building -> Loss Calculation -> Backward Pass -> Optimizer Step`.
* **Output Contract:** It yields a lightweight `StepResult` dictionary containing the loss values and detached metrics. It **does not** average metrics or interact with loggers.

### 2. `BatchIterators` (The Data Shield)
Iterators decouple data sampling from the optimization loop. They act as the adapter between the Replay Buffer and the Learner.
* **Responsibility:** They pull pinned memory from the buffer, move it asynchronously to the GPU (`non_blocking=True`), and handle all algorithm-specific batching logic.
* **Examples:** `SingleBatchIterator` (yields once for DQN/Imitation) vs. `PPOEpochIterator` (shuffles and yields mini-batches over multiple epochs).

### 3. `TargetBuilders` (The Math)
Target builders are pure, stateless mathematical functions. They are named semantically based on the math they perform, not the algorithm that uses them.
* **Contract:** `batch = builder(batch, predictions)`. They accept a dictionary, compute RL math (like $n$-step returns or MCTS value scalarization), and attach the new tensors to the dictionary. 
* **Note:** They **do not** handle neural network distribution projections (like C51 or MuZero categorical logic); that is strictly the job of the `LossPipeline`.

### 4. `Callbacks` (The Event System)
All side-effects and algorithm-specific control flows are handled via isolated callbacks injected into the Learner. Callbacks communicate with the outside world purely via injected `Callable` functions.
* **`PPOEarlyStoppingCallback`:** Reads the KL divergence and raises an `EarlyStopIteration` exception to gracefully break the Learner's epoch loop.
* **`PriorityUpdaterCallback`:** Hands PER priorities back to the Trainer via an injected queue/function.
* **`WeightBroadcastCallback`:** Broadcasts the `state_dict` to remote Actors.

### 5. `Factory` (The Assembly Layer)
The factory wires the components together using the **Registry Pattern**. 
* It looks up the requested algorithm in the `AGENT_REGISTRY` (e.g., `@register_agent("ppo")`), grabs the specific Iterators, Loss Pipelines, and Callbacks, and assembles the `UniversalLearner`. This keeps the factory completely flat and free of `if/else` chains.

---

## 🔄 The Data Flow Lifecycle

1. **Setup:** The Trainer calls `build_universal_learner`. The factory pulls the specific components from the registry.
2. **Ingestion:** The Trainer hands a `BatchIterator` to `learner.step(iterator)`.
3. **Processing:** The Learner loops over the iterator. Tensors are routed through the `TargetBuilderPipeline` and then the `LossPipeline`.
4. **Optimization:** The Learner dynamically iterates over the loss dictionary, calling `.backward()` and stepping the respective optimizers (enabling multi-optimizer architectures like Actor-Critic).
5. **Events & Telemetry:** Callbacks fire at designated hook points (`on_backward_end`, `on_step_end`). The Learner yields a detached metrics dictionary back to the Trainer for logging.