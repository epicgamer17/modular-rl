from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING, Union
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from modules.agent_nets.modular import ModularAgentNetwork

if TYPE_CHECKING:
    from agents.learners.target_builders import BaseTargetBuilder
    from losses.losses import LossPipeline


@dataclass
class StepResult:
    """Container for a single optimization update."""

    loss: Union[torch.Tensor, Dict[str, torch.Tensor]]
    loss_dict: Dict[str, float]
    priorities: Optional[torch.Tensor] = None
    predictions: Dict[str, torch.Tensor] = field(default_factory=dict)
    targets: Dict[str, torch.Tensor] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


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
        stats=None,
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
        stats=None,
        **kwargs,
    ) -> None:
        """Fired after each sub-batch optimization (forward + backward + step)."""
        pass  # pragma: no cover

    def on_training_step_end(
        self,
        learner: UniversalLearner,
        stats=None,
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
        stats=None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_backward_end(
                learner=learner,
                step_result=step_result,
                stats=stats,
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

    def on_training_step_end(
        self,
        learner: UniversalLearner,
        stats=None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_training_step_end(learner=learner, stats=stats)


class UniversalLearner:
    """
    Algorithm-agnostic learner that orchestrates the optimization loop.

    Acts as a pure data-processing pipe: it does not own a replay buffer,
    does not know about batch sizes, and does not sample data. The caller
    (trainer/orchestrator) provides a batch_iterator — any iterable that
    yields batch dicts. The learner processes each batch through:
    forward → targets → loss → backward → step.

    Algorithm-specific behaviour is injected via:
    - TargetBuilder (DQN Bellman targets vs. passthrough)
    - LossPipeline (which loss modules to run)
    - Callbacks (target network sync, KL early stopping, noise reset)
    - The batch_iterator itself (single batch vs. PPO multi-epoch)
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
        optimizer: Optional[Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]] = None,
        lr_scheduler: Optional[Union[Any, Dict[str, Any]]] = None,
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
        self.lr_scheduler = lr_scheduler

        self.training_step = 0
        self.callbacks = CallbackList(callbacks)

    def _preprocess_observation(self, states: Any) -> torch.Tensor:
        """Converts states to torch tensors on the correct device.

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

    def step(
        self, batch_iterator: Iterable[Dict[str, Any]], stats=None
    ) -> Optional[Dict[str, Any]]:
        """Processes all batches from the iterator through the optimization loop.

        The learner is blind to where the data comes from. For DQN/Imitation,
        the iterator yields exactly one batch. For PPO, it yields shuffled
        mini-batches across multiple epochs. The learner just processes until
        the iterator is exhausted or a callback raises EarlyStopIteration.

        Args:
            batch_iterator: Any iterable yielding batch dicts.
            stats: Optional stat tracker for logging metrics.

        Returns:
            Dictionary of loss statistics from the last processed batch,
            or None if no batches were processed.
        """
        start_time = time.time()
        last_result: Optional[StepResult] = None
        batches_processed = 0

        self.callbacks.on_step_begin(learner=self, iterator=batch_iterator)

        try:
            for batch in batch_iterator:
                # 1. Gradient Clearing
                if isinstance(self.optimizer, dict):
                    for opt in self.optimizer.values():
                        opt.zero_grad(set_to_none=True)
                else:
                    self.optimizer.zero_grad(set_to_none=True)

                # 2. Forward + Targets + Loss
                result = self.compute_step_result(batch=batch, stats=stats)

                # 3. Backward
                if isinstance(result.loss, dict):
                    for loss_tensor in result.loss.values():
                        loss_tensor.backward()
                else:
                    result.loss.backward()

                # 4. Fire on_backward_end (e.g. PPO KL early stopping, metrics)
                self.callbacks.on_backward_end(
                    learner=self, step_result=result, stats=stats
                )

                # 5. Gradient Clipping
                if self.clipnorm > 0:
                    if isinstance(self.optimizer, dict):
                        for opt in self.optimizer.values():
                            opt_params = [p for group in opt.param_groups for p in group["params"]]
                            clip_grad_norm_(opt_params, self.clipnorm)
                    else:
                        clip_grad_norm_(self.agent_network.parameters(), self.clipnorm)

                # 6. Weight Update
                if isinstance(self.optimizer, dict):
                    for opt in self.optimizer.values():
                        opt.step()
                else:
                    self.optimizer.step()

                # 7. LR Scheduler Step
                if self.lr_scheduler is not None:
                    if isinstance(self.lr_scheduler, dict):
                        for sched in self.lr_scheduler.values():
                            sched.step()
                    else:
                        self.lr_scheduler.step()

                # 8. Fire on_optimizer_step_end (e.g. noisy net reset)
                self.callbacks.on_optimizer_step_end(learner=self)

                # 9. Fire on_step_end
                self.callbacks.on_step_end(
                    learner=self,
                    predictions=result.predictions,
                    targets=result.targets,
                    loss_dict=result.loss_dict,
                    priorities=result.priorities,
                    stats=stats,
                    batch=batch,
                    meta=result.meta,
                )

                last_result = result
                batches_processed += 1

        except EarlyStopIteration:
            pass  # Callback requested early stop (e.g. PPO KL divergence)

        self.training_step += 1

        # Fire on_training_step_end (e.g. target network sync)
        self.callbacks.on_training_step_end(learner=self, stats=stats)

        if stats is not None and batches_processed > 0:
            duration = time.time() - start_time
            if duration > 0:
                stats.append("learner_fps", batches_processed / duration)

        self._maybe_clear_mps_cache()

        if last_result is None:
            return None

        total_loss_val = sum(l.item() for l in last_result.loss.values()) if isinstance(last_result.loss, dict) else float(last_result.loss.item())

        return self._prepare_stats(
            last_result.loss_dict, total_loss_val
        )

    def compute_step_result(self, batch: Dict[str, Any], stats=None) -> StepResult:
        """Pure data-processing pipe: forward → targets → loss.

        1. Forward Pass (Predictions)
        2. Build Targets (via TargetBuilder or passthrough)
        3. Run Loss Pipeline
        """
        # 1. Predictions
        predictions = self.agent_network.learner_inference(batch)

        # 2. Targets
        if self.target_builder is not None:
            targets = self.target_builder.build_targets(
                batch, predictions, self.agent_network
            )
        else:
            targets = batch

        # 3. Context and PER weights
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
            gradient_scales=None,
        )

        # Prepare detached copies for callbacks/logging safety
        if hasattr(predictions, "_asdict"):
            preds_dict = predictions._asdict()
        elif isinstance(predictions, dict):
            preds_dict = predictions
        else:
            preds_dict = vars(predictions)

        targs_dict = targets if isinstance(targets, dict) else {}

        return StepResult(
            loss=loss,
            loss_dict=loss_dict,
            priorities=priorities,
            predictions={
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in preds_dict.items()
            },
            targets={
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in targs_dict.items()
            },
        )

    @property
    def clipnorm(self) -> float:
        """Maximum gradient norm for clipping. 0 means no clipping."""
        return self.config.clipnorm

    def save_checkpoint(self, path: str):
        """Saves learner state (network weights, optimizer state, training step)."""
        checkpoint = {
            "agent_network": self.agent_network.state_dict(),
            "training_step": self.training_step,
        }
        if self.optimizer is not None:
            if isinstance(self.optimizer, dict):
                checkpoint["optimizer"] = {k: v.state_dict() for k, v in self.optimizer.items()}
            else:
                checkpoint["optimizer"] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, dict):
                checkpoint["lr_scheduler"] = {k: v.state_dict() for k, v in self.lr_scheduler.items()}
            else:
                checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Loads learner state from path."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.agent_network.load_state_dict(checkpoint["agent_network"])
        self.training_step = checkpoint.get("training_step", 0)

        if "optimizer" in checkpoint and self.optimizer is not None:
            if isinstance(self.optimizer, dict):
                for k, v in self.optimizer.items():
                    if k in checkpoint["optimizer"]:
                        v.load_state_dict(checkpoint["optimizer"][k])
            else:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint and self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, dict):
                for k, v in self.lr_scheduler.items():
                    if k in checkpoint["lr_scheduler"]:
                        v.load_state_dict(checkpoint["lr_scheduler"][k])
            else:
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

    def preprocess(self, observation: Any) -> torch.Tensor:
        """Preprocesses observation for network input.

        Args:
            observation: Raw observation.

        Returns:
            Preprocessed tensor on the correct device.
        """
        return self._preprocess_observation(observation)
