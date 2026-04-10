import torch
from core import PipelineComponent
from core import Blackboard


class ResetNoiseComponent(PipelineComponent):
    """Resets noisy network parameters after each execution."""

    def __init__(self, agent_network: torch.nn.Module):
        self.agent_network = agent_network

    def execute(self, blackboard: Blackboard) -> None:
        if hasattr(self.agent_network, "reset_noise"):
            self.agent_network.reset_noise()
