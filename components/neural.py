import torch
from typing import Optional, Dict, Any, TYPE_CHECKING
from core import PipelineComponent
from core import Blackboard

if TYPE_CHECKING:
    from modules.agent_nets.base import BaseAgentNetwork
    from learner.losses.shape_validator import ShapeValidator
    from actors.action_selectors.policy_sources import BasePolicySource


class ForwardPassComponent(PipelineComponent):
    """
    Executes the main neural network forward pass.
    Reads inputs from Blackboard batch and writes outputs to Blackboard predictions.
    """
    def __init__(self, agent_network: 'BaseAgentNetwork', shape_validator: Optional['ShapeValidator'] = None):
        self.agent_network = agent_network
        self.shape_validator = shape_validator

    def execute(self, blackboard: Blackboard) -> None:
        """
        Runs learner_inference on the data dictionary.
        Optimizes memory layout for throughput before the pass.
        """
        # OPTIMIZATION: Convert convolutional observations to channels_last for Tensor Cores
        # Only if the device is CUDA and it's a 4D tensor.
        for k, v in blackboard.data.items():
            if torch.is_tensor(v) and v.ndim == 4 and v.device.type == "cuda":
                blackboard.data[k] = v.to(memory_format=torch.channels_last)

        predictions = self.agent_network.learner_inference(
            blackboard.data, shape_validator=self.shape_validator
        )
        
        # Merge predictions safely into the blackboard
        blackboard.predictions.update(predictions)


class BurnInComponent(PipelineComponent):
    """
    Handles LSTM/RNN initialization using a Burn-In period.
    Runs a torch.no_grad() forward pass on the first L steps of a sequence
    to generate a pristine hidden state before handing the remaining K steps
    to the main ForwardPassComponent.
    """
    def __init__(self, agent_network: 'BaseAgentNetwork', burn_in_steps: int):
        self.agent_network = agent_network
        self.burn_in_steps = burn_in_steps

    def execute(self, blackboard: Blackboard) -> None:
        """
        Splits the data observations into burn-in and train segments.
        Generates initial hidden states using the burn-in segment.
        """
        if self.burn_in_steps <= 0:
            return
            
        # 1. Extract observations from data
        # We assume standard sequence data format [B, T, ...]
        obs = blackboard.data.get("observations")
        if obs is None or not torch.is_tensor(obs):
            return
            
        if obs.ndim < 2 or obs.shape[1] <= self.burn_in_steps:
            return

        # 2. Split into burn-in and training segments
        # L = burn_in_steps
        burn_in_obs = obs[:, :self.burn_in_steps]
        # Re-inject the remaining steps back into the data for the main ForwardPassComponent
        blackboard.data["observations"] = obs[:, self.burn_in_steps:]

        # 3. Generate pristine hidden states
        # We assume the network has a method or we can use the world model's encoder
        with torch.no_grad():
            # Use inference mode for throughput
            with torch.inference_mode():
                # This assumes the network can handle sequence burn-in
                # and returns the final hidden state in the network_state
                burn_in_result = self.agent_network.burn_in_inference(burn_in_obs)
                
                # 4. Inject into the data for the ForwardPassComponent to pick up
                blackboard.data["hidden_state"] = burn_in_result.network_state


class StateInjectionComponent(PipelineComponent):
    """
    Manages explicit injection of hidden states into the sequence batch.
    Used when the replay buffer stores the continuous hidden states (e.g. from actors)
    rather than relying on burn-in computation.
    """
    def __init__(self, state_key: str = "hidden_state"):
        self.state_key = state_key

    def execute(self, blackboard: Blackboard) -> None:
        """
        Ensures that recurrent states are retrieved and correctly mapped into
        the batch for the ForwardPassComponent.
        """
        # E.g. Check for blackboard.data['network_state'], format it, 
        # validate it against shape constraints, and ensure it requires_grad 
        # if using Truncated BPTT.
        pass


class ActorInferenceComponent(PipelineComponent):
    """Runs ``obs_inference`` via a ``PolicySource`` and writes the result.

    Writes ``predictions["inference_result"]`` (an ``InferenceResult``).
    Uses ``torch.inference_mode`` internally for throughput.
    """

    def __init__(self, policy_source: 'BasePolicySource'):
        self.policy_source = policy_source

    def execute(self, blackboard: Blackboard) -> None:
        obs = blackboard.data["observations"]
        info = blackboard.meta.get("info", {})

        with torch.inference_mode():
            result = self.policy_source.get_inference(obs=obs, info=info)

        blackboard.predictions["inference_result"] = result
