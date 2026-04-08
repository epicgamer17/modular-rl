"""Pipeline components that replace the old Callback system.

Each former callback is now a standard PipelineComponent placed directly
in the BlackboardEngine recipe list. No hooks, no dispatcher — just
sequential execution.
"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

import torch

from learner.pipeline.base import PipelineComponent

if TYPE_CHECKING:
    from learner.core import Blackboard
    from modules.agent_nets.modular import ModularAgentNetwork


class TargetNetworkSyncComponent(PipelineComponent):
    """Syncs target network weights at a configured interval."""

    def __init__(
        self,
        target_network: ModularAgentNetwork,
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


class ResetNoiseComponent(PipelineComponent):
    """Resets noisy network parameters after each execution."""

    def __init__(self, agent_network: torch.nn.Module):
        self.agent_network = agent_network

    def execute(self, blackboard: Blackboard) -> None:
        self.agent_network.reset_noise()


class MetricEarlyStopComponent(PipelineComponent):
    """Signals the engine to stop when a metric exceeds a threshold.

    Sets ``blackboard.meta["stop_execution"] = True`` so that
    ``BlackboardEngine`` breaks out of the batch iteration loop.
    """

    def __init__(self, metric_key: str = "approx_kl", threshold: float = 0.015):
        self.metric_key = metric_key
        self.threshold = threshold

    def execute(self, blackboard: Blackboard) -> None:
        loss_dict = {
            k: v.item()
            for k, v in blackboard.losses.items()
            if k != "total_loss" and torch.is_tensor(v)
        }
        val = loss_dict.get(self.metric_key)
        if val is not None and val > self.threshold:
            blackboard.meta["stop_execution"] = True


class PriorityBufferUpdateComponent(PipelineComponent):
    """Pushes computed priorities back into the replay buffer."""

    def __init__(self, priority_update_fn: Callable[..., None]):
        self.priority_update_fn = priority_update_fn

    def execute(self, blackboard: Blackboard) -> None:
        priorities = blackboard.meta.get("priorities")
        if priorities is None:
            return

        batch = blackboard.batch
        priorities_np = priorities.detach().cpu().numpy()
        ids = batch.get("ids")
        self.priority_update_fn(batch["indices"], priorities_np, ids=ids)


class BetaScheduleComponent(PipelineComponent):
    """Advances the PER beta schedule and updates the buffer."""

    def __init__(self, set_beta_fn: Callable[[float], None], per_beta_schedule: Any):
        self.set_beta_fn = set_beta_fn
        self.per_beta_schedule = per_beta_schedule

    def execute(self, blackboard: Blackboard) -> None:
        self.set_beta_fn(self.per_beta_schedule.get_value())


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


class EpsilonScheduleComponent(PipelineComponent):
    """Steps the epsilon-greedy exploration schedule."""

    def __init__(self, epsilon_schedule: Any):
        self.epsilon_schedule = epsilon_schedule

    def execute(self, blackboard: Blackboard) -> None:
        self.epsilon_schedule.step()


class MPSCacheClearComponent(PipelineComponent):
    """Clears the MPS cache at a configured interval."""

    def __init__(self, device: torch.device, interval: int = 100):
        self.device = device
        self.interval = interval
        self._step_counter = 0

    def execute(self, blackboard: Blackboard) -> None:
        if self.interval <= 0:
            return

        self._step_counter += 1
        is_mps = (
            self.device.type == "mps"
            if isinstance(self.device, torch.device)
            else self.device == "mps"
        )
        if is_mps and self._step_counter % self.interval == 0:
            torch.mps.empty_cache()
