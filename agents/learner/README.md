# 🧠 Learners: The Universal GPU Engine

This module contains the core optimization engine for the reinforcement learning pipeline. It is designed with a strict **Component-Centric, Data-Oriented Architecture**. 

There are no algorithm-specific subclasses (e.g., `PPOLearner` or `MuZeroLearner` do not exist). Instead, a single `UniversalLearner` acts as a pure, stateless mathematical pipeline. We build complex RL algorithms (like AlphaZero, Rainbow, or PPO) strictly by composing pure functional units via **Factories**.

## 🏗️ Core Contracts & Philosophy

### 1. The `Universal T` Contract
The entire learner pipeline is strictly vectorized over the sequence/time dimension.
* Every tensor flowing through the Loss Pipeline must be of shape `[Batch, Time, ...]`.
* For sequence models (MuZero), `T` is the unroll horizon.
* For single-step models (PPO, DQN, Imitation), `T = 1`. The network, target builders, and loss modules seamlessly handle this via native PyTorch broadcasting.

### 2. `UniversalLearner` (The Engine)
The Learner is a blind GPU engine. It knows nothing about Replay Buffers, environments, RL dynamics, or telemetry dashboards.
* **Input Contract:** It accepts an `Iterable` that yields raw dictionary batches.
* **Execution Contract:** It processes data through a rigid sequence: `Forward Pass -> Target Building -> Loss Execution -> Multi-Optimizer Backward Pass -> Step`.
* **Output Contract:** It yields a lightweight dictionary containing loss values and decoupled metrics. It does not manage loggers or shared mutable state.

### 3. `TargetBuilders` (The Middleware Pipeline)
Target builders are responsible for the **MDP Math** (Markov Decision Process logic like Bellman shifts, GAE, or MCTS extraction). They operate as a composable middleware chain modifying a `current_targets` dictionary.
We split builders into two strict categories to avoid monolithic classes:
* **Generators:** Extract or compute mathematical targets (e.g., `MCTSExtractor`, `TemporalDifferenceBuilder`).
* **Modifiers:** Align, pad, mask, or format tensors without changing their RL meaning (e.g., `SequencePadder`, `SingleStepFormatter`, `UniversalMaskBuilder`).
* **Explicit Anchoring:** Builders *never* dynamically guess dimensions by looping over dictionaries. Shapes are strictly derived from anchors like `batch["actions"]`.

### 4. `Callbacks` (The Event System)
We rely on isolated callbacks for all side-effects.
* **`PriorityUpdaterCallback`:** Syncs computed TD/MSE priorities back to the replay buffer.
* **`TargetNetworkSyncCallback`:** Handles EMA or hard syncing for target networks.
* **`MetricEarlyStopCallback`:** Stops epochs dynamically based on emitted metrics (e.g., KL divergence limits).

---

## 🚀 Adding a New Algorithm
**Do not subclass the Learner.** To create a new algorithm (e.g., "Rainbow + Search" or "Dreamer"):
1. Create a factory function in the `registries/` folder.
2. Snap together the appropriate `TargetBuilder` generators and modifiers.
3. Instantiate pure `BaseLoss` modules with the correct string keys.
4. Pass the assembled pipeline to the `UniversalLearner`.