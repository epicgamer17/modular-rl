🧠 RainbowZero Core Modules
This directory contains the neural network component library for the RainbowZero framework.

The architecture has been radically refactored from script-like "God-classes" into a Declarative Component Library. By enforcing strict tensor contracts and absolute separation of concerns, this engine can dynamically assemble networks for PPO, Recurrent PPO, Rainbow DQN, MuZero, and Stochastic MuZero entirely via configuration, without a single if algorithm == ... branch in the forward passes.

🧭 Core Philosophy
Dumb Components, Smart Routers: Individual neural network layers (Backbones, Heads) do not know anything about RL algorithms, Time dimensions, or sequence lengths. They just do raw math on batches of tensors. The routing logic (AgentNetwork, WorldModel) handles all temporal logic and dictionary merging.

Data Flow over Control Flow: We avoid if hasattr(...) and if stochastic: inside forward passes. Instead, we use the Strategy Pattern (e.g., DeterministicDynamics vs StochasticDynamics) and explicit component instantiation.

No Magic: We do not dynamically guess shapes during the forward pass. Math is deterministic. Expected shapes are calculated during __init__ and enforced via rigorous assertions.

🏗️ Architecture Overview
The system is split into two primary routers and three categories of Lego-block components.

The Routers (in models/)
1. AgentNetwork (The Actor/Learner Core)
The absolute center of the framework. It acts as the Switchboard between the RL System and the PyTorch Sub-modules.

Owns: The Representation backbone (Encoder), the Memory Core (RNNs), and all Behavior Heads (Policy, Value, Q-Values).

Role: Routes raw observations to the latent space, optionally passes them to the World Model to simulate the future, and maps latents to behaviors.

2. WorldModel (The Environment Simulator)
A self-contained "Simulator in a Box" that can be pre-trained, frozen, or transferred.

Owns: The DynamicsPipeline and all Environment Heads (Reward, To-Play, Continuation).

Role: Strictly simulates environment physics. It takes a latent state and an action, predicts the next latent state, and predicts the semantics of the environment. It does not know about policy or value.

The Components
backbones/ (Feature Extractors): Pure PyTorch networks (ResNets, MLPs, Transformers). They map an input tensor to a feature tensor. They do not output semantics.

heads/ (Semantic Predictors): Terminal layers. They map a feature tensor to a specific RL concept (e.g., ValueHead, PolicyHead). All heads strictly return a standardized HeadOutput.

embeddings/ (Action Translation): Handles the translation of RL actions (discrete or continuous) into dense vectors or spatial planes, and fuses them into latent states.

📜 Strict Contracts
To ensure dynamic assembly works without crashing, all modules must adhere to the following strict contracts.

1. The Flat-Batching Contract
PyTorch spatial layers (Convs, Linear) do not natively understand the Time dimension (B,T,...).

Backbones and Heads: Must strictly expect purely flat batches: (B∗,Features). They never reshape sequences.

The Router: The AgentNetwork explicitly intercepts sequences, flattens them tensor.flatten(0, 1), passes them through spatial backbones, and unflattens them .view(B, T, -1) only at the boundary of a Memory Core (RNN) or just before returning the final loss dictionary.

2. The Recurrent State Contract
A recurrent state is strictly a flat dictionary of PyTorch Tensors: Dict[str, Tensor].

No nested dictionaries. No tuples. No lists.

If an LSTM outputs a tuple (h, c), it must be packed into the dictionary as {"lstm_h": h, "lstm_c": c} by the Backbone before being returned to the Router.

This ensures that state merging via ** unpacking and state batching is perfectly uniform and branchless.

3. The HeadOutput Contract
Every BaseHead subclass must return a HeadOutput dataclass:

Python
@dataclass
class HeadOutput:
    training_tensor: torch.Tensor     # The raw logits/values for the Learner's loss function
    inference_tensor: Optional[Any]   # The argmax / PyTorch Distribution for the Actor
    state: Dict[str, torch.Tensor]    # Updated memory state (empty {} if stateless)
Masking: Invalid action masking is handled inside the PolicyHead. It masks the logits to -1e8 before returning the training_tensor, completely isolating the Loss Pipeline from the concept of legal moves.

4. The Action Fusion Contract
Dynamics models do not accept vague "contexts". Actions are fused explicitly.

ActionFusion takes a (latent, action). It embeds the action via ActionEncoder and concatenates/adds it to the latent before passing it to the raw dynamics backbones.

🛠️ Adding a New Algorithm (Example)
Because of this modularity, adding complex algorithmic features requires zero changes to the core routing logic.

Want to add an Auxiliary Observation Reconstruction Loss to PPO?

Create ObservationReconstructionHead in heads/observation.py.

Add "reconstruction": ObservationHeadConfig(...) to your PPO configuration's heads dictionary.

You're done. The AgentNetwork will dynamically instantiate the head, route the memory features to it, and output "reconstruction" in the final dictionary for the Loss Pipeline.

Want to swap standard MuZero to Stochastic MuZero?

Set stochastic = True in the config.

The WorldModel dynamically instantiates the StochasticDynamics pipeline instead of the DeterministicDynamics pipeline.

The forward passes automatically generate afterstates and chance codes.

🧹 Technical Debt Zero
No __init__(self, config) in components: Components accept strict, typed kwargs. They do not depend on massive global configuration objects.

No target_latents in the forward pass: The network only computes predictions. Target construction is strictly delegated to the Learner/TargetBuilder.

No manual .device tracking: Devices are resolved using the native dummy-buffer pattern.