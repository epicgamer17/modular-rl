from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from utils.schedule import create_schedule, Schedule
from replay_buffers.modular_buffer import ModularReplayBuffer
from modules.agent_nets.modular import ModularAgentNetwork

if TYPE_CHECKING:
    from agents.learners.target_builders import BaseTargetBuilder
    from losses.losses import LossPipeline


@dataclass
class StepResult:
    """Container for a single optimization update."""

    loss: torch.Tensor
    loss_dict: Dict[str, float]
    priorities: Optional[torch.Tensor] = None
    predictions: Dict[str, torch.Tensor] = field(default_factory=dict)
    targets: Dict[str, torch.Tensor] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


class Callback(ABC):
    """Learner callback interface."""

    def on_step_end(
        self,
        learner: UniversalLearner,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_dict: Dict[str, float],
        stats=None,
        **kwargs,
    ) -> None:
        pass  # pragma: no cover


class CallbackList:
    """Lightweight callback collection wrapper."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def on_step_end(
        self,
        learner: UniversalLearner,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_dict: Dict[str, float],
        stats=None,
        **kwargs,
    ) -> None:
        for callback in self.callbacks:
            callback.on_step_end(
                learner=learner,
                predictions=predictions,
                targets=targets,
                loss_dict=loss_dict,
                stats=stats,
                **kwargs,
            )


class UniversalLearner:
    """
    Concrete learner that orchestrates the optimization loop.
    Decouples algorithm logic via TargetBuilder and LossPipeline components.
    """

    def __init__(
        self,
        config,
        agent_network: ModularAgentNetwork,
        device: torch.device,
        num_actions: int,
        observation_dimensions: Tuple[int, ...],
        observation_dtype: torch.dtype,
        target_builder: Optional[BaseTargetBuilder] = None,
        loss_pipeline: Optional[LossPipeline] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        replay_buffer: Optional[ModularReplayBuffer] = None,
        lr_scheduler: Optional[Any] = None,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.config = config
        self.agent_network = agent_network
        self.device = device
        self.num_actions = num_actions
        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype

        self.target_builder = target_builder
        self.loss_pipeline = loss_pipeline
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.lr_scheduler = lr_scheduler

        self.training_step = 0
        self.callbacks = CallbackList(callbacks)
        self.schedules: Dict[str, Schedule] = {}
        self._init_schedules()

    def _init_schedules(self):
        if (
            hasattr(self.config, "per_beta_schedule")
            and self.config.per_beta_schedule is not None
        ):
            self.schedules["per_beta"] = create_schedule(self.config.per_beta_schedule)

        if (
            hasattr(self.config, "epsilon_schedule")
            and self.config.epsilon_schedule is not None
        ):
            self.schedules["epsilon"] = create_schedule(self.config.epsilon_schedule)

    def _preprocess_observation(self, states: Any) -> torch.Tensor:
        """
        Converts states to torch tensors on the correct device.
        Adds batch dimension if input is a single observation.
        """
        if torch.is_tensor(states):
            if states.device == self.device and states.dtype == torch.float32:
                prepared_state = states
            else:
                prepared_state = states.to(self.device, dtype=torch.float32)
        else:
            np_states = np.array(states, copy=False)
            prepared_state = torch.tensor(
                np_states, dtype=torch.float32, device=self.device
            )

        if prepared_state.ndim == len(self.observation_dimensions):
            prepared_state = prepared_state.unsqueeze(0)

        return prepared_state

    def step(self, stats=None) -> Optional[Dict[str, Any]]:
        """Performs one learner update cycle."""
        if self.replay_buffer.size < self.min_buffer_size:
            return None

        start_time = time.time()
        last_result: Optional[StepResult] = None

        for _ in range(self.training_iterations):
            batch = self.replay_buffer.sample()

            # 1. Gradient Clearing
            self.optimizer.zero_grad(set_to_none=True)

            # 2. Compute Step Result (orchestrates forward, targets, and loss)
            result = self.compute_step_result(batch=batch, stats=stats)

            # 3. Backward
            result.loss.backward()

            # 4. Gradient Clipping
            if self.clipnorm > 0:
                clip_grad_norm_(self.agent_network.parameters(), self.clipnorm)

            # 5. Weight Update
            self.optimizer.step()

            # 6. LR Scheduler Step
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.after_optimizer_step(batch=batch, step_result=result, stats=stats)

            # 7. Priority Updates for PER
            if result.priorities is not None:
                priorities = result.priorities
                self.replay_buffer.update_priorities(
                    batch["indices"], priorities, ids=batch.get("ids")
                )

            last_result = result

        self._step_schedules()
        self.training_step += 1

        if stats is not None:
            duration = time.time() - start_time
            if duration > 0:
                fps = (self.config.minibatch_size * self.training_iterations) / duration
                stats.append("learner_fps", fps)

        self._maybe_clear_mps_cache()

        if last_result is None:
            return None

        self.callbacks.on_step_end(
            learner=self,
            predictions=last_result.predictions,
            targets=last_result.targets,
            loss_dict=last_result.loss_dict,
            stats=stats,
            batch=batch,
            meta=last_result.meta,
        )
        return self._prepare_stats(
            last_result.loss_dict, float(last_result.loss.item())
        )

    def compute_step_result(self, batch: Dict[str, Any], stats=None) -> StepResult:
        """
        Generic optimization step:
        1. Forward Pass (Predictions)
        2. Build Targets
        3. Run Loss Pipeline
        """
        # 1. Predictions
        # The agent_network is expected to handle internal preprocessing if needed.
        predictions = self.agent_network.learner_inference(batch)

        # 2. Targets
        # The target_builder computes what the predictions should have ideally been.
        targets = self.target_builder.build_targets(
            batch, predictions, self.agent_network
        )

        # 3. Contextual and PER data
        # We pass the raw batch as context to the loss pipeline for masks, etc.
        context = batch
        weights = batch.get("weights")
        if weights is not None and torch.is_tensor(weights):
            weights = weights.to(self.device).float()

        # 4. Loss calculation
        loss, loss_dict, priorities = self.loss_pipeline.run(
            predictions=predictions,
            targets=targets,
            context=context,
            weights=weights,
            # gradient_scales defaults to None, let pipeline detect unroll depth
            gradient_scales=None,
        )

        # Prepare for StepResult
        preds_dict = (
            predictions._asdict() if hasattr(predictions, "_asdict") else predictions
        )
        targs_dict = targets._asdict() if hasattr(targets, "_asdict") else targets

        return StepResult(
            loss=loss,
            loss_dict=loss_dict,
            priorities=priorities,
            # Detach for callbacks/logging safety
            predictions={k: v.detach().cpu() for k, v in preds_dict.items()},
            targets={k: v.detach().cpu() for k, v in targs_dict.items()},
        )

    @property
    def min_buffer_size(self) -> int:
        return self.config.min_replay_buffer_size

    @property
    def training_iterations(self) -> int:
        return self.config.training_iterations

    @property
    def clipnorm(self) -> float:
        return self.config.clipnorm

    def _step_schedules(self) -> None:
        for name, schedule in self.schedules.items():
            schedule.step()
            if name == "per_beta":
                self.replay_buffer.set_beta(schedule.get_value())
            elif name == "epsilon":
                self.action_selector.epsilon = schedule.get_value()

    def save_checkpoint(self, path: str):
        """
        Saves learner state (network weights, optimizer state, training step).
        """
        checkpoint = {
            "agent_network": self.agent_network.state_dict(),
            "training_step": self.training_step,
        }
        if hasattr(self, "optimizer"):
            checkpoint["optimizer"] = self.optimizer.state_dict()
        if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Loads learner state from path.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.agent_network.load_state_dict(checkpoint["agent_network"])
        self.training_step = checkpoint.get("training_step", 0)

        if "optimizer" in checkpoint and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if (
            "lr_scheduler" in checkpoint
            and hasattr(self, "lr_scheduler")
            and self.lr_scheduler is not None
        ):
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    def _maybe_clear_mps_cache(self) -> None:
        if isinstance(self.device, torch.device):
            is_mps = self.device.type == "mps"
        else:
            is_mps = self.device == "mps"
        if is_mps and self.training_step % 100 == 0:
            torch.mps.empty_cache()

    def _prepare_stats(
        self, loss_dict: Dict[str, float], total_loss: float
    ) -> Dict[str, float]:
        stats = dict(loss_dict)
        stats["loss"] = total_loss
        return stats

    def after_optimizer_step(
        self, batch: Dict[str, Any], step_result: StepResult, stats=None
    ) -> None:
        """Optional hook for subclasses or logging. Automatically resets Noisy layers if present."""
        # Reset online network noise
        if hasattr(self.agent_network, "reset_noise"):
            self.agent_network.reset_noise()

    def preprocess(self, observation: Any) -> torch.Tensor:
        """
        Preprocesses observation for network input.

        Args:
            observation: Raw observation.

        Returns:
            Preprocessed tensor on the correct device.
        """
        return self._preprocess_observation(observation)
