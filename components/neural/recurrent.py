import torch
from typing import TYPE_CHECKING
from core import PipelineComponent
from core import Blackboard

if TYPE_CHECKING:
    from modules.agent_nets.base import BaseAgentNetwork


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
