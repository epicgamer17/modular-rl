from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from replay_buffers.utils import update_per_beta


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
        learner: "BaseLearner",
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_dict: Dict[str, float],
        stats=None,
        **kwargs,
    ) -> None:
        pass


class CallbackList:
    """Lightweight callback collection wrapper."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def on_step_end(
        self,
        learner: "BaseLearner",
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


class BaseLearner(ABC):
    """
    Generic learner optimization loop:
    Sample -> Forward/Loss -> Backward -> Update Priorities -> Callbacks
    """

    def __init__(
        self,
        config,
        model,
        device,
        num_actions,
        observation_dimensions,
        observation_dtype,
        callbacks: Optional[List[Callback]] = None,
    ):
        self.config = config
        self.model = model
        self.device = device
        self.num_actions = num_actions
        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype
        self.training_step = 0
        self.callbacks = CallbackList(callbacks)

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

        if prepared_state.ndim == 0:
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
            result = self.compute_step_result(batch=batch, stats=stats)

            self.optimizer.zero_grad(set_to_none=True)
            result.loss.backward()

            if self.clipnorm > 0:
                clip_grad_norm_(self.model.parameters(), self.clipnorm)

            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.after_optimizer_step(batch=batch, step_result=result, stats=stats)

            if result.priorities is not None:
                priorities = result.priorities
                if isinstance(priorities, torch.Tensor):
                    priorities = priorities.detach().cpu().numpy()
                self.replay_buffer.update_priorities(
                    batch["indices"], priorities, ids=batch.get("ids")
                )

            last_result = result

        self._update_per_beta()
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

    @property
    def min_buffer_size(self) -> int:
        return self.config.min_replay_buffer_size

    @property
    def training_iterations(self) -> int:
        return self.config.training_iterations

    @property
    def clipnorm(self) -> float:
        return self.config.clipnorm

    def _update_per_beta(self) -> None:
        self.replay_buffer.set_beta(
            update_per_beta(
                self.replay_buffer.beta,
                self.config.per_beta_final,
                self.config.training_steps,
                self.config.per_beta,
            )
        )

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
        """Optional hook for subclasses."""
        return None

    @abstractmethod
    def compute_step_result(self, batch: Dict[str, Any], stats=None) -> StepResult:
        """Compute forward pass and loss for one sampled batch."""
        raise NotImplementedError
