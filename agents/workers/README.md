Worker Engine Architecture
This module contains the isolated, highly concurrent execution engines (Actors) responsible for interacting with environments, computing MCTS/Network inferences, and streaming mathematically complete trajectories to the central Replay Buffer.

🎯 High-Level Goals
Maximize Throughput: Stream fixed-size chunks (e.g., 200 steps) to prevent GPU starvation, natively supporting distributed architectures like Ape-X and MuZero Unplugged.

Component-Centric Composition: Eradicate inheritance-based combinatorial explosions (e.g., PettingZooPufferDAggerActor). Actors are built by composing Adapters, Policy Sources, and State Managers.

Environment Isolation: Neural networks and Actors should never know whether they are playing Gym, PettingZoo, or PufferLib. All idiosyncrasies are caught at the Adapter boundary.

Unified On/Off-Policy Streaming: Support both overlapping N-step unrolls (MuZero/Rainbow) and strict epoch-based bootstraps (PPO) through a single collection interface.

🧠 Core Philosophy
Fail-Fast: If a component receives bad data, it crashes at the front door. We use strict assertions for tensor shapes, dtypes, and math requirements (e.g., ensuring a bootstrap_value is present for truncated PPO sequences).

Data-Oriented (Batched by Default): Operations act on PyTorch tensors with a leading [Batch, ...] dimension. Actors do not loop over environments; they pass the full batch directly to the PolicySource.

Stateless Workers (Where Possible): Actors do not track global steps, epsilon decay, or training iterations. They receive configurations from the Orchestrator (Trainer) via WorkerPayloads and execute blindly.

No Gradient Tape in Workers: All environment stepping and data collection loops are strictly decorated with @torch.inference_mode() to prevent catastrophic RAM leaks.

📜 Strict Contracts & Interfaces
1. EnvironmentAdapter (The Bridge)
Adapters are the defensive boundary protecting the math from the real world.

Contract: step() and reset() MUST return PyTorch tensors of shape [Batch, ...].

Type Casting: Adapters explicitly cast discrete environment arrays to torch.float32 (or appropriate types) to prevent Char vs Float convolution crashes.

Auto-Resetting: Actors never call .reset() after the initial setup. The Adapter automatically resets terminated environments and splices the fresh obs and infos into the returned batch.

Masking: Legal move masks must be returned as [B, A] boolean tensors in the info dictionary.

2. SequenceManager (The Sliding Window Engine)
Replaces the outdated concept of waiting for an episode to finish. It manages localized memory for each vector environment.

Off-Policy Contract (Rainbow/MuZero): Uses the Overlap Strategy. When a chunk boundary is reached, it flushes steps 0 to (ChunkSize - N) to the central buffer, and retains the final N steps in local memory to seamlessly start the next chunk. This guarantees mathematically unbroken N-step targets.

On-Policy Contract (PPO): Uses the Bootstrap Strategy. Flushes all steps in the chunk, zeroes out local memory, and strictly requires the Actor to provide a bootstrap_value tensor for the unfinished boundary states to prevent Generalized Advantage Estimation (GAE) corruption.

3. Execution Units (The Actors)
Actors do not subclass environment types. They are defined purely by their Task.

RolloutActor: The workhorse. Calls collect(N). Steps the adapter, feeds the SequenceManager, and streams chunks to the central buffer.

EvaluatorActor: Evaluates until an episode is done. Has no buffer and no SequenceManager. Handles routing to hardcoded test_agents via the player_id info key. Tracks scores per-player.

ReanalyzeActor: Contains no environment. Queries the Replay Buffer for historical [B, T, ...] states, flattens them, runs MCTS, and overwrites the buffer with fresh target policies.

DAggerActor / NFSPActor: Specialty task actors that inject multiple networks (Student + Expert / RL + Average) to label data dynamically during the environment loop.

4. Multiprocessing & Orchestration
WorkerPayload: Cross-process communication uses strict Dataclasses. No raw dictionaries or tuples.

Throughput Tracking: Throughput (FPS) is calculated globally by the Orchestrator (Trainer/Executor) based on the size of the returned payloads, completely decoupling the measurement from the Actor's internal loops.

Dynamic Adapters: The Orchestrator resolves the correct EnvironmentAdapter class dynamically based on the game config (e.g., vectorized, multi-agent) before sending it to the Actor.

🛠️ Developer Notes & Usage Guide
When to use overlap_length vs flush_at_end:

If setting up PPO: Configure the RolloutActor with overlap_length=0 and flush_at_end=True. You must pass the value network's estimate of the final state to the SequenceManager's flush trigger.

If setting up Rainbow / MuZero: Configure the RolloutActor with overlap_length=N (where N is the unroll/target length) and flush_at_end=False.

Adding a New Environment Library:
Do not touch the Actors. Create a new adapter (e.g., DeepMindControlAdapter) in agents/environments/adapters.py. Ensure it outputs [B, ...], handles auto-resets internally, and registers it in BaseTrainer._get_adapter_class().

Handling Multi-Agent Termination:
Zero-sum environments (like Tic-Tac-Toe) require both the winner (+1) and loser (-1) to receive their final rewards. Your Adapter implementation must ensure it captures the terminal -1 state for the drained agent before fully resetting the environment, otherwise EvaluatorActor scores will sum to 0.

Updating Hyperparameters:
If an Actor needs an updated Epsilon or MCTS Temperature, the Trainer must push a hyperparams dict via actor.update_parameters(). Do not rely on internal step counters inside the Actor's Action Selectors, as multiprocessing desyncs them.