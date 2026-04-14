import torch
from typing import Set, Dict, Any
from core import PipelineComponent, Blackboard
from core.contracts import Key

class ResetNoiseComponent(PipelineComponent):
    """Resets noisy network parameters after each execution."""

    def __init__(self, agent_network: torch.nn.Module):
        self.agent_network = agent_network

    @property
    def requires(self) -> Set[Key]:
        return set()

    @property
    def provides(self) -> Set[Key]:
        return set()

    def validate(self, blackboard: Blackboard) -> None:
        pass

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        if hasattr(self.agent_network, "reset_noise"):
            self.agent_network.reset_noise()
        return {}
