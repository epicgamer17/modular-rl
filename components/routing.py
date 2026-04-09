import torch
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
from core import PipelineComponent
from core import Blackboard

if TYPE_CHECKING:
    from modules.agent_nets.base import BaseAgentNetwork


class TargetNetworkSyncComponent(PipelineComponent):
    """Syncs target network weights at a configured interval."""

    def __init__(
        self,
        target_network: 'BaseAgentNetwork',
        sync_interval: int,
        soft_update: bool = False,
        ema_beta: float = 0.99,
    ):
        self.target_network = target_network
        self.sync_interval = sync_interval
        self.soft_update = soft_update
        self.ema_beta = ema_beta
        self._step_counter = 0

    def execute(self, blackboard: Blackboard) -> None:
        self._step_counter += 1
        if self.sync_interval <= 0 or self._step_counter % self.sync_interval != 0:
            return

        from modules.utils import get_clean_state_dict

        source_network = blackboard.meta.get("agent_network")
        if source_network is None:
            return

        with torch.no_grad():
            clean_state = get_clean_state_dict(source_network)
            if self.soft_update:
                target_state = self.target_network.state_dict()
                beta = self.ema_beta
                for k, v in clean_state.items():
                    if k not in target_state:
                        continue
                    if target_state[k].is_floating_point():
                        target_state[k].mul_(beta).add_(v.detach(), alpha=1.0 - beta)
                    else:
                        target_state[k].copy_(v.detach())
            else:
                self.target_network.load_state_dict(clean_state, strict=False)


class WeightBroadcastComponent(PipelineComponent):
    """Broadcasts network weights to remote workers."""

    def __init__(
        self,
        agent_network: torch.nn.Module,
        weight_broadcast_fn: Callable[..., None],
    ):
        self.agent_network = agent_network
        self.weight_broadcast_fn = weight_broadcast_fn

    def execute(self, blackboard: Blackboard) -> None:
        self.weight_broadcast_fn(self.agent_network.state_dict())


class ResetNoiseComponent(PipelineComponent):
    """Resets noisy network parameters after each execution."""

    def __init__(self, agent_network: torch.nn.Module):
        self.agent_network = agent_network

    def execute(self, blackboard: Blackboard) -> None:
        if hasattr(self.agent_network, "reset_noise"):
            self.agent_network.reset_noise()


class MultiplexerComponent(PipelineComponent):
    """
    Placeholder for multi-agent routing.
    Routes blackboard data to specific agent sub-pipelines.
    """
    def execute(self, blackboard: Blackboard) -> None:
        pass
