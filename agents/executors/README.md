# Executors: The Concurrency Backend

The `executors` module is the concurrency engine of the modular RL library. 

**An Executor's sole responsibility is Inter-Process Communication (IPC).** It acts as a strict router that manages the lifecycle of distributed workers and moves data payloads across process, thread, or network boundaries. 

**Executors know absolutely nothing about Reinforcement Learning.** They do not know what MCTS is, they do not calculate episode scores, and they do not process trajectories.

## Why this Separation of Concerns? (The N x M Problem)
By strictly separating **Algorithm Logic** (Trainers) from **Concurrency Logic** (Executors), we avoid a combinatorial explosion of code. 

If Trainers handled their own multiprocessing, we would need a `RayPPOTrainer`, `TorchMPPPOTrainer`, and `LocalPPOTrainer`. By keeping Executors pure, any algorithm (PPO, MuZero, Rainbow) can seamlessly run on any hardware backend (Local debugging, Torch Multiprocessing, or a distributed Ray cluster) without modifying a single line of algorithm code.

---

## The Contract

All Executors inherit from `BaseExecutor` and must strictly adhere to the Payload Data Contracts.

### The Payload Types
The Executor only accepts and returns strict data classes. Duck-typing and raw tuples are forbidden.
1. **`UpdatePayload`**: Sent *to* workers. Contains `state_dict` updates or hyperparameter changes.
2. **`TaskRequest`**: Sent *to* workers. Contains an Enum command (e.g., `TaskType.COLLECT`, `TaskType.EVALUATE`) and a requested `batch_size`.
3. **`WorkerPayload`**: Received *from* workers. Contains standardized `metrics` (dicts for logging) and `data` (e.g., trajectory chunks).

### The Interface
* `launch(worker_cls, args, num_workers)`: Initializes the compute pool.
* `broadcast_params(payload: UpdatePayload)`: Distributes weights globally.
* `dispatch(request: TaskRequest)`: Sends commands to the pool.
* `gather() -> List[WorkerPayload]`: Non-blocking retrieval of completed work.
* `stop()`: Cleans up processes, queues, and memory.

---

## Usage: How Trainers Interact with Executors

Trainers treat the Executor as an asynchronous black box. 

```python
# 1. Setup
executor = TorchMPExecutor()
executor.launch(RolloutActor, actor_args, num_workers=8)

# 2. Update Weights
executor.broadcast_params(UpdatePayload(weights=latest_network_weights))

# 3. Request Work
executor.dispatch(TaskRequest(task_type=TaskType.COLLECT, batch_size=2048))

# 4. Wait & Gather (The Trainer handles the RL logic, not the executor)
payloads = []
while len(payloads) < target_batches:
    new_results = executor.gather()
    payloads.extend(new_results)
    
# 5. Process
metrics = calculate_trainer_stats(payloads)