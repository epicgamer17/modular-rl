from __future__ import annotations

from typing import Callable, Dict, Any, Iterable, List, Optional, TYPE_CHECKING

import torch
from utils.telemetry import (
    add_latent_visualization_metric,
    append_metric,
    set_metric,
)

if TYPE_CHECKING:
    from agents.learners.base import StepResult, UniversalLearner
    from modules.agent_nets.modular import ModularAgentNetwork
from abc import ABC, abstractmethod


class EarlyStopIteration(Exception):
    """Raised by callbacks to break the batch iteration loop early."""

    pass


class Callback(ABC):
    """Learner callback interface with event hooks."""

    def on_step_begin(
        self,
        learner: UniversalLearner,
        iterator: Iterable[Dict[str, Any]],
    ) -> None:
        """Fired at the very beginning of the learner step, before fetching batches."""
        pass  # pragma: no cover

    def on_backward_end(
        self,
        learner: UniversalLearner,
        step_result: StepResult,
    ) -> None:
        """Fired after loss.backward() but before optimizer.step().

        Raise EarlyStopIteration to break the batch iteration loop.
        """
        pass  # pragma: no cover

    def on_optimizer_step_end(
        self,
        learner: UniversalLearner,
    ) -> None:
        """Fired exactly after optimizer.step() and optional lr_scheduler.step()."""
        pass  # pragma: no cover

    def on_step_end(
        self,
        learner: UniversalLearner,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_dict: Dict[str, float],
        **kwargs,
    ) -> None:
        """Fired after each sub-batch optimization (forward + backward + step)."""
        pass  # pragma: no cover

    def on_training_step_end(
        self,
        learner: UniversalLearner,
        metrics: Dict[str, Any],
    ) -> None:
        """Fired after the full training step (all iterations complete)."""
        pass  # pragma: no cover


class CallbackList:
    """Lightweight callback collection wrapper."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def on_step_begin(
        self,
        learner: UniversalLearner,
        iterator: Iterable[Dict[str, Any]],
    ) -> None:
        for callback in self.callbacks:
            callback.on_step_begin(learner=learner, iterator=iterator)

    def on_backward_end(
        self,
        learner: UniversalLearner,
        step_result: StepResult,
    ) -> None:
        for callback in self.callbacks:
            callback.on_backward_end(
                learner=learner,
                step_result=step_result,
            )

    def on_optimizer_step_end(
        self,
        learner: UniversalLearner,
    ) -> None:
        for callback in self.callbacks:
            callback.on_optimizer_step_end(learner=learner)

    def on_step_end(
        self,
        learner: UniversalLearner,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_dict: Dict[str, float],
        **kwargs,
    ) -> None:
        for callback in self.callbacks:
            callback.on_step_end(
                learner=learner,
                predictions=predictions,
                targets=targets,
                loss_dict=loss_dict,
                **kwargs,
            )

    def on_training_step_end(
        self,
        learner: UniversalLearner,
        metrics: Dict[str, Any],
    ) -> None:
        for callback in self.callbacks:
            callback.on_training_step_end(learner=learner, metrics=metrics)




class TargetNetworkSyncCallback(Callback):
    """Syncs target network weights at the end of each training step."""

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

    def on_training_step_end(
        self,
        learner: UniversalLearner,
        metrics: Dict[str, Any],
    ) -> None:
        if self.sync_interval > 0 and learner.training_step % self.sync_interval != 0:
            return

        from modules.utils import get_clean_state_dict

        with torch.no_grad():
            clean_state = get_clean_state_dict(learner.agent_network)
            if self.soft_update:
                target_state = self.target_network.state_dict()
                ema_beta = self.ema_beta
                for k, v in clean_state.items():
                    if k not in target_state:
                        continue
                    if target_state[k].is_floating_point():
                        target_state[k].mul_(ema_beta).add_(
                            v.detach(), alpha=1.0 - ema_beta
                        )
                    else:
                        target_state[k].copy_(v.detach())
            else:
                self.target_network.load_state_dict(clean_state, strict=False)


class ResetNoiseCallback(Callback):
    """Resets noisy network parameters after every optimizer step."""

    def on_optimizer_step_end(
        self,
        learner,
    ) -> None:
        learner.agent_network.reset_noise()





class PPOEarlyStoppingCallback(Callback):
    """Early stops the optimization loop if KL divergence exceeds target_kl."""

    def __init__(self, target_kl: float, key: str = "approx_kl"):
        assert target_kl > 0, f"target_kl must be positive, got {target_kl}"
        self.target_kl = target_kl
        self.key = key

    def on_backward_end(
        self,
        learner,
        step_result: StepResult,
    ) -> None:
        """Checks KL divergence against target threshold."""
        assert self.key in step_result.loss_dict, (
            f"Key '{self.key}' missing from StepResult.loss_dict. "
            f"Available keys: {list(step_result.loss_dict.keys())}. "
            "Ensure the LossPipeline is propagating this metric from context."
        )
        kl = step_result.loss_dict[self.key]
        if kl > 1.5 * self.target_kl:
            append_metric(step_result.meta.setdefault("metrics", {}), "ppo_early_stop", 1.0)
            raise EarlyStopIteration(f"KL divergence {kl:.4f} > 1.5 * {self.target_kl}")


class PriorityUpdaterCallback(Callback):
    """Updates Prioritized Experience Replay (PER) buffer priorities via a callback."""

    def __init__(
        self,
        priority_update_fn: Callable,
        set_beta_fn: Callable,
        per_beta_schedule: Any,
    ):
        """
        Initializes the PriorityUpdaterCallback.

        Args:
            priority_update_fn: A callable that accepts (indices, priorities, ids).
            set_beta_fn: A callable to update PER beta (accepts float).
            per_beta_schedule: A schedule for updating PER beta.
        """
        self.priority_update_fn = priority_update_fn
        self.set_beta_fn = set_beta_fn
        self.per_beta_schedule = per_beta_schedule

    def on_step_end(
        self,
        learner: UniversalLearner,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_dict: Dict[str, float],
        **kwargs,
    ) -> None:
        batch = kwargs["batch"]
        priorities = kwargs["priorities"]

        ids = batch.get("ids")
        priorities_np = priorities.detach().cpu().numpy()
        self.priority_update_fn(batch["indices"], priorities_np, ids=ids)

    def on_training_step_end(
        self,
        learner: UniversalLearner,
        metrics: Dict[str, Any],
    ) -> None:
        self.set_beta_fn(self.per_beta_schedule.get_value())


class WeightBroadcastCallback(Callback):
    """Broadcasts network weights to remote workers/executors."""

    def __init__(self, weight_broadcast_fn: Callable[[Dict[str, Any]], None]):
        """
        Initializes the WeightBroadcastCallback.

        Args:
            weight_broadcast_fn: A callable that accepts the state_dict of the agent network.
        """
        self.weight_broadcast_fn = weight_broadcast_fn

    def on_training_step_end(
        self,
        learner: UniversalLearner,
        metrics: Dict[str, Any],
    ) -> None:
        """Broadcast weights at the end of the full training step (after all iterations/batches)."""
        self.weight_broadcast_fn(learner.agent_network.state_dict())


class EpsilonGreedySchedulerCallback(Callback):
    """Syncs target network weights at the end of each training step."""

    # TODO: should we have the init like this or should the learner store the target_agent_network too?
    def __init__(self, epsilon_schedule):
        self.epsilon_schedule = epsilon_schedule

    def on_training_step_end(
        self,
        learner,
        metrics: Dict[str, Any],
    ) -> None:
        self.epsilon_schedule.step()


class MPSCacheClearCallback(Callback):
    """Clears the MPS cache periodically outside the learner core loop."""

    def __init__(self, interval: int = 100):
        self.interval = interval

    def on_training_step_end(
        self,
        learner: UniversalLearner,
        metrics: Dict[str, Any],
    ) -> None:
        if self.interval <= 0:
            return

        device = learner.device
        is_mps = device.type == "mps" if isinstance(device, torch.device) else device == "mps"
        if is_mps and learner.training_step % self.interval == 0:
            torch.mps.empty_cache()
