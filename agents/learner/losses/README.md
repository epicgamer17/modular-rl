# 📉 Losses: The Pure Math Engine

This directory contains the highly optimized, vectorized loss execution pipeline. The defining rule of this module is **Separation of Concerns**: Losses execute pure tensor math. They know absolutely nothing about Reinforcement Learning, Bellman equations, or algorithm names.

## 📜 The Architectural Contracts

### Contract 1: Pure Functions & No Shared State
Loss functions must not mutate shared dictionaries or state. 
* A loss function takes exactly two inputs: `predictions: Dict[str, Tensor]` and `targets: Dict[str, Tensor]`.
* It returns exactly two outputs: `elementwise_loss: Tensor [B, T]` and `metrics: Dict[str, float]`.
* If a loss needs to log an auxiliary metric (like `approx_kl`), it calculates it internally and returns it in the `metrics` dictionary.

### Contract 2: No MDP Logic
A loss module or representation must never accept a `reward`, `discount`, or `done` flag. 
* If a target distribution needs to be shifted by a reward (e.g., Distributional RL / C51), that Bellman math must happen in the `TargetBuilder`. 
* The Loss Pipeline expects the target tensor to arrive perfectly formatted and ready for standard metric comparison (Cross Entropy, MSE, Cosine Similarity).

### Contract 3: Explicit Key Injection
We do not write `PPOValueLoss` or `MuZeroRewardLoss`. We use mathematically named losses (e.g., `ActionValueLoss`, `ClippedSurrogateLoss`, `CosineSimilarityLoss`) or the generic `BaseLoss`.
* The semantic meaning of a loss is defined entirely by the keys passed to its constructor. 
* Example: `BaseLoss(pred_key="rewards", target_key="rewards", loss_fn=F.mse_loss)` acts as a Reward Loss without needing a custom class.

---

## 🧮 Representations (The Geometry Layer)
Representations handle the geometric transformation of tensors (e.g., converting a scalar to a probability distribution). They are completely decoupled from algorithms and are grouped by mathematical structure:

* **`DiscreteSupportRepresentation`:** The base class for all grid-based math. Holds `vmin`, `vmax`, and `support`.
    * **`TwoHotRepresentation`:** Projects a single scalar onto the two nearest bins (used natively by MuZero).
    * **`DistributionalRepresentation`:** Exposes a `project_onto_grid` API to align shifted probability distributions (used by Rainbow/C51).
* **`ClassificationRepresentation`:** Standard one-hot encoding for discrete integer classes.
* **`ExponentialBucketsRepresentation`:** A *Decorator* that warps targets into log-space, delegating to an inner grid representation, enabling exponential scaling for any algorithm seamlessly.

## 🚦 The Loss Pipeline Execution
The `LossPipeline` class is a loop-free, vectorized runner.
1. **Validates:** Enforces the `[B, T]` Universal contract via `ShapeValidator`.
2. **Executes:** Iterates through the registered `BaseLoss` modules.
3. **Reduces:** Multiplies the `[B, T]` element-wise losses by `weights` (PER) and `gradient_scales` (BPTT scaling).
4. **Masks:** Multiplies by the boolean mask and averages strictly over the valid mathematical transitions.
5. **Priorities:** Delegates to a pure `BasePriorityComputer` to extract buffer priorities (e.g., Root MSE error for MuZero, Max TD error for DQN).