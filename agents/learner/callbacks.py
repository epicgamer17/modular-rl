from __future__ import annotations

from typing import Callable, Dict, Any, Iterable, List, Optional, TYPE_CHECKING

import torch
from utils.telemetry import (
    add_latent_visualization_metric,
    append_metric,
    set_metric,
)

if TYPE_CHECKING:
    from agents.learner.base import StepResult, UniversalLearner
    from modules.models.agent_network import AgentNetwork
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
        target_network: AgentNetwork,
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

        from modules.utils import update_target_network

        tau = (1.0 - self.ema_beta) if self.soft_update else 1.0
        update_target_network(learner.agent_network, self.target_network, tau=tau)


class ResetNoiseCallback(Callback):
    """Resets noisy network parameters after every optimizer step."""

    def __init__(self, target_network: Optional[Any] = None):
        self.target_network = target_network

    def on_optimizer_step_end(
        self,
        learner,
    ) -> None:
        learner.agent_network.reset_noise()
        self.target_network.reset_noise()


class MetricEarlyStopCallback(Callback):
    """Agnostic early stopping callback triggered by any metric threshold."""

    def __init__(self, metric_key: str = "approx_kl", threshold: float = 0.015):
        self.metric_key = metric_key
        self.threshold = threshold

    def on_backward_end(
        self,
        learner,
        step_result: StepResult,
    ) -> None:
        val = step_result.loss_dict.get(self.metric_key)
        if val is not None and val > self.threshold:
            print(f"Early stopping triggered by {self.metric_key}: {val}")
            raise EarlyStopIteration()


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
    """Steps the epsilon-greedy exploration schedule at the end of each training step."""

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
        is_mps = (
            device.type == "mps"
            if isinstance(device, torch.device)
            else device == "mps"
        )
        if is_mps and learner.training_step % self.interval == 0:
            torch.mps.empty_cache()
