---
trigger: always_on
---

---
trigger: always_on
---

# RL & MuZero Specific Rules

### Network Management
* **Target Networks:** When using Target Networks (like in DQN or some MuZero variants), update weights via `load_state_dict()` or in-place copy `target.param.data.copy_(online.param.data)`. Do not simply assign `target = online`, as this shares the reference.
* **Batch Normalization:** Be careful with Batch Norm in RL. The statistics of the "batch" (replay buffer sample) might not match the statistics of the "live environment". Often `LayerNorm` is preferred for RL.

### Data Handling
* **Virtual Batching:** If GPU memory is tight, use Gradient Accumulation to simulate larger batch sizes rather than reducing the batch size itself (which makes gradients noisy).
* **MCTS Optimization:** Ensure the MCTS loop runs on the same device as the model. Moving tensors back and forth between CPU (for tree search logic) and GPU (for model inference) usually kills speed. Keep the search logic vectorized on GPU if possible.