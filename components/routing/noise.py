import torch
from typing import Dict, Any, Set
from core import PipelineComponent, Blackboard
from core.contracts import Key

class ResetNoiseComponent(PipelineComponent):
    """Resets noisy network parameters after each execution."""

    def __init__(self, agent_network: torch.nn.Module):
        self.agent_network = agent_network
        self._requires = set()
        self._provides = {}

    @property
    def requires(self) -> Set[Key]:
        return self._requires

    @property
    def provides(self) -> Dict[Key, str]:
        return self._provides

    def validate(self, blackboard: Blackboard) -> None:
        """No inputs to validate; this component is a side-effect-only utility."""
        pass

    def execute(self, blackboard: Blackboard) -> Dict[str, Any]:
        if hasattr(self.agent_network, "reset_noise"):
            self.agent_network.reset_noise()
        return {}
