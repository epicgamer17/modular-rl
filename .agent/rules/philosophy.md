---
trigger: always_on
---

THIS RULE SHOULD ALWAYS BE USED WHEN PLANNING, AND AFTER IMPLEMENTING YOU SHOULD CHECK THAT ALL CHANGES FOLLOW THIS PHILOSPHY AND RULE. 

Project Architecture & Rules
1. Core Philosophy
This framework is built on the principle of Strict Separation of Concerns and Perfect Polymorphism.
Deep Reinforcement Learning systems collapse when neural network math bleeds into game logic, or when optimization math bleeds into tree search logic.

The Blind Learner: The Learner (Optimization) must be completely blind to the neural network's architecture (LSTM, ResNet, Transformer). It asks for raw math, receives it, and computes the loss.

The Blind Actor/Tree: The MCTS Tree and the Actor must treat the network's memory (network_state) as an Opaque Token. They store it and pass it back to the network without ever looking inside it.

Game Logic Isolation: The Neural Network is a pure mathematical function. It does not know the rules of the game. Action masking and legal move filtering happen strictly outside the network (in the Action Selectors or the Search Tree).

⚠️ LIVING DOCUMENT: If new folders, core classes, or foundational APIs are added to this project, this RULES.md file must be updated to reflect their architectural boundaries.

2. Directory Structure & Domain Boundaries
modules/ (The Compute Graph)
Domain: Pure PyTorch Neural Networks.

Contains: AgentNetworks, Backbones, Heads, WorldModels.

Rule: Classes here know nothing about RL environments, Replay Buffers, or loss functions. They strictly turn tensors into other tensors.

learner/ (The Optimization Logic)
Domain: The Loss Pipeline and Optimizer.

Contains: PPOLearner, RainbowLearner, MuZeroLearner.

Rule: The Learner orchestrates backpropagation. It must never contain PyTorch graph routing (e.g., for t in range(T): ...). It asks the AgentNetwork for an unrolled output and feeds it to the losses/.

actors/action_selectors/ (The Bridge)
Domain: Translating Math into Actions.

Contains: CategoricalSelector, ArgmaxSelector, MCTSSelector.

Rule: This is where game rules meet network math. Action masking happens here. It extracts the action from a PyTorch Distribution.

data/ (The Fact Store)
Domain: High-performance, multi-processed data storage.

Contains: ModularReplayBuffer, Writers, Samplers, Processors.

Rule: The buffer stores immutable facts (uint8 images). Mathematical targets (like N-step returns or GAE) are calculated on the fly during sample() via Processors to prevent stale data during Reanalyze.

search/ (The Imagination)
Domain: CPU-bound MCTS and hypothetical rollouts.

Rule: The search tree lives entirely on the CPU. It interacts with the GPU strictly by batching opaque states and passing them to the AgentNetwork.

3. The 3 Stages of Preprocessing
Preprocessing is notoriously leaky. We strictly divide it into three distinct phases to protect RAM and GPU VRAM:

Game Preprocessing (Environment Wrappers): Resizing, grayscale, frame stacking. Output is a raw NumPy array (e.g., uint8).

Tensor Routing (Action Selectors & Buffer Samplers): Converts NumPy arrays to PyTorch Tensors, adds the batch dimension (unsqueeze(0)), and moves to the correct device. Does not change dtype.

Neural Preprocessing (Network Backbones): Happens strictly inside the network on the GPU. Casts uint8 to float32 etc.

4. Core Class Contracts
BaseAgentNetwork (The Composer)
The absolute center of the framework. It acts as the Switchboard between the RL System and the PyTorch Sub-modules.

Public API (The 3 Domains):

obs_inference(obs) -> InferenceOutput: Used by Actor/MCTS for real-world root states.

hidden_state_inference(hidden_state, action) -> InferenceOutput: Used by MCTS for latent stepping.

learner_inference(batch) -> Dict[str, Tensor]: Used by the Learner to get raw math tensors across a batch/sequence.

get_learner_contract() -> Dict[str, Type[SemanticType]]: Used by ForwardPassComponent to determine providing keys and their structured semantic types.

Contract: * InferenceOutput returns semantic objects (Expected Values, PyTorch Distribution objects). It never returns raw logits.

Raw Tensors returned by learner_inference must have their semantic types declared via get_learner_contract(). 

Must pack and unpack all sub-module RNN states into the network_state dictionary (The Opaque Token).

WorldModel (The Physics Engine)
Public API: encode, step, unroll_physics.

Contract: A pure math block. It returns its own internal RNN states via a head_state or similar field. It relies on the BaseAgentNetwork to store and feed these states back to it.

ActionSelector
Public API: select_action(agent_network, obs, info)

Contract: Responsible for calling obs_inference. Must apply action_mask directly to the Distribution object's logits and repackage it before sampling to prevent illegal moves.

Learner
Public API: step(batch)

Contract: Cannot loop over network modules. Must call learner_inference(batch), unpack the UnrollOutput, and route the raw tensors to the LossPipeline.

5. Golden Rules & Anti-Patterns
No "Dummy Tensors": Never return torch.empty(0) or torch.zeros() to satisfy a dataclass field you don't use. Use None. If an algorithm accidentally touches a field it shouldn't, a TypeError is safer than a silent broadcast bug.

No PyTorch Object Passing in Loss Functions: Loss functions take raw Tensors (logits and targets). They do not take PyTorch Distribution objects (which can hide numerically unstable softmax operations).

No deepcopy: Never use copy.deepcopy() on observations inside the Replay Buffer or Actor loops. Trust the environment's NumPy arrays or use .copy() for a 100x speedup.

Action Masking is Ruthless: If an action mask is applied, the illegal logit must be set to -inf. If Softmax or Gumbel noise is applied after an initial mask, the mask must be re-applied to explicitly set the illegal probability to 0.0 before sampling.

No Python Loops in Samplers: Replay buffer processors (like N-step target builders) must use vectorized tensor operations. Looping over a batch in Python will starve the GPU.

6. Structured Contracts & DAG Validation
Semantic Types with Structure: All contracts use parameterized types: ValueEstimate[Scalar], Policy[Categorical(bins=51)], Reward[Quantile(n=32)]. String-based distribution mapping is FORBIDDEN.

Automated Discovery: Components MUST NOT manually define output contracts based on config strings. They must query the AgentNetwork's get_learner_contract() API.

Build-Time Validation: The BlackboardEngine enforces contract consistency before execution starts via a 4-stage validation:
1. Dependency Resolution (Path existence)
2. Semantic Compatibility (Type subclassing)
3. Representation Consistency (Metadata/Parameters like vmin, vmax, bins matching)
4. Shape Integrity (Dimensionality and structure-specific shape checks)