from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import torch
from torch.nn.utils import clip_grad_norm_


if TYPE_CHECKING:
    from agents.learner.target_builders import BaseTargetBuilder
    from agents.learner.losses import LossPipeline
    from agents.learner.callbacks import Callback
    from modules.models.agent_network import AgentNetwork

from agents.learner.callbacks import (
    CallbackList,
    EarlyStopIteration,
    MPSCacheClearCallback,
)
from utils.telemetry import finalize_metrics


@dataclass
class StepResult:
    """Container for a single optimization update."""

    loss: Dict[str, torch.Tensor]
    loss_dict: Dict[str, float]
    priorities: Optional[torch.Tensor] = None
    predictions: Dict[str, torch.Tensor] = field(default_factory=dict)
    targets: Dict[str, torch.Tensor] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


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
        agent_network: AgentNetwork,
        device: torch.device,
        num_actions: int,
        observation_dimensions: Tuple[int, ...],
        observation_dtype: torch.dtype,
        target_builder: Optional[BaseTargetBuilder] = None,
        loss_pipeline: Optional[LossPipeline] = None,
        optimizer: Optional[
            Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]
        ] = None,
        lr_scheduler: Optional[Union[Any, Dict[str, Any]]] = None,
        callbacks: Optional[List[Callback]] = None,
        clipnorm: Optional[float] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        **kwargs,
    ):
        self.agent_network = agent_network
        self.device = device
        self.num_actions = num_actions
        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype

        self.target_builder = target_builder
        self.loss_pipeline = loss_pipeline
        self.clipnorm = clipnorm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm or clipnorm

        # Normalize optimizers and schedulers into dictionaries
        if isinstance(optimizer, dict):
            self.optimizers = optimizer
        elif optimizer is not None:
            self.optimizers = {"default": optimizer}
        else:
            self.optimizers = {}

        if isinstance(lr_scheduler, dict):
            self.lr_schedulers = lr_scheduler
        elif lr_scheduler is not None:
            self.lr_schedulers = {"default": lr_scheduler}
        else:
            self.lr_schedulers = {}

        self.training_step = 0
        callback_list = list(callbacks or [])
        callback_list.append(MPSCacheClearCallback())
        self.callbacks = CallbackList(callback_list)

    def step(
        self, batch_iterator: Iterable[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Standard optimization loop with Gradient Accumulation support.

        The learner is blind to where the data comes from. For DQN/Imitation,
        the iterator yields exactly one batch. For PPO, it yields shuffled
        mini-batches across multiple epochs. The learner just processes until
        the iterator is exhausted or a callback raises EarlyStopIteration.
        """
        accum_steps = self.gradient_accumulation_steps

        current_result: Optional[StepResult] = None
        yielded_current_result = False

        self.callbacks.on_step_begin(learner=self, iterator=batch_iterator)
        t_last = time.perf_counter()

        # Ensure we start with clean gradients
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)

        try:
            for step_idx, batch in enumerate(batch_iterator):
                # 1. Forward Pass + Targets + Loss
                result = self.compute_step_result(batch=batch)
                current_result = result
                yielded_current_result = False

                # 2. Backward Pass (Routed to specific optimizers)
                # We iterate over unique optimizers to avoid redundant backward passes on aggregated losses
                for opt_key, total_loss in result.loss.items():
                    # CRITICAL: Scale the loss so the accumulated gradients are averaged!
                    loss_tensor = total_loss / accum_steps

                    # Backward with retain_graph as multiple optimizers might share parameters
                    loss_tensor.backward(retain_graph=len(result.loss) > 1)

                # 3. Accumulation Boundary
                if (step_idx + 1) % accum_steps == 0:

                    # Optional Gradient Clipping (Global across all parameters)
                    if self.max_grad_norm is not None and self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.agent_network.parameters(), self.max_grad_norm
                        )

                    # Step and clear
                    for opt in self.optimizers.values():
                        opt.step()
                        opt.zero_grad(set_to_none=True)

                    # Fire callbacks that happen after weight update
                    self.callbacks.on_backward_end(learner=self, step_result=result)

                    # LR Scheduler Step
                    for sched in self.lr_schedulers.values():
                        sched.step()

                    # Fire on_optimizer_step_end (e.g. noisy net reset)
                    self.callbacks.on_optimizer_step_end(learner=self)

                # 4. Fire on_step_end
                self.callbacks.on_step_end(
                    learner=self,
                    predictions=result.predictions,
                    targets=result.targets,
                    loss_dict=result.loss_dict,
                    priorities=result.priorities,
                    batch=batch,
                    meta=result.meta,
                )

                yielded_current_result = True

                # 5. Throughput Metrics
                t_now = time.perf_counter()
                dt = t_now - t_last
                t_last = t_now

                # Fetch B, T from any valid prediction
                any_pred = next(
                    p for p in result.predictions.values() if torch.is_tensor(p)
                )
                B, T = any_pred.shape[:2]

                result.loss_dict["learner_throughput"] = (B * T) / dt if dt > 0 else 0
                yield self._build_step_metrics(result)

        except EarlyStopIteration:
            if current_result is not None and not yielded_current_result:
                yield self._build_step_metrics(current_result)

        self.training_step += 1

        # Fire on_training_step_end (e.g. target network sync)
        training_step_metrics: Dict[str, Any] = {}
        self.callbacks.on_training_step_end(learner=self, metrics=training_step_metrics)
        finalized_training_step_metrics = finalize_metrics(training_step_metrics)
        if finalized_training_step_metrics:
            yield {"metrics": finalized_training_step_metrics}

    def compute_step_result(self, batch: Dict[str, Any]) -> StepResult:
        """Pure data-processing pipe: forward -> targets -> loss.

        1. Forward Pass (Predictions)
        2. Build Targets (via TargetBuilder or passthrough)
        3. Run Loss Pipeline
        4. Priories
        """
        # 1. Predictions
        predictions = self.agent_network.learner_inference(batch)

        # 2. Targets (Strict Delegation)
        # The TargetBuilder is the ONLY source of truth for the LossPipeline.
        targets: Dict[str, torch.Tensor] = {}
        if self.target_builder is not None:
            self.target_builder.build_targets(
                batch=batch,
                predictions=predictions,
                network=self.agent_network,
                current_targets=targets,  # Mutated in-place!
            )

        # 3. PER weights
        weights = batch.get("weights")
        if weights is not None and torch.is_tensor(weights):
            weights = weights.to(self.device).float()

        # 4. Loss calculation
        loss, loss_dict, priorities = self.loss_pipeline.run(
            predictions=predictions,
            targets=targets,
            weights=weights,
            gradient_scales=targets.get("gradient_scales"),
        )

        # Prepare detached copies for callbacks/logging safety
        return StepResult(
            loss=loss,
            loss_dict=loss_dict,
            priorities=priorities,
            predictions={
                k: v.detach().cpu()
                for k, v in predictions.items()
                if torch.is_tensor(v)
            },
            targets={
                k: v.detach().cpu() for k, v in targets.items() if torch.is_tensor(v)
            },
            meta={"metrics": batch.get("metrics", {})},
        )

    def state_dict(self) -> Dict[str, Any]:
        """Returns learner state in standard PyTorch dictionary form."""
        return {
            "agent_network": self.agent_network.state_dict(),
            "training_step": self.training_step,
            "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
            "lr_schedulers": {k: v.state_dict() for k, v in self.lr_schedulers.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Loads learner state from a standard PyTorch dictionary."""
        self.agent_network.load_state_dict(state["agent_network"])
        self.training_step = state.get("training_step", 0)

        if "optimizers" in state:
            for k, v in self.optimizers.items():
                if k in state["optimizers"]:
                    v.load_state_dict(state["optimizers"][k])
        # Backward compatibility for old checkpoints
        elif "optimizer" in state:
            if "default" in self.optimizers:
                self.optimizers["default"].load_state_dict(state["optimizer"])

        if "lr_schedulers" in state:
            for k, v in self.lr_schedulers.items():
                if k in state["lr_schedulers"]:
                    v.load_state_dict(state["lr_schedulers"][k])
        # Backward compatibility for old checkpoints
        elif "lr_scheduler" in state:
            if "default" in self.lr_schedulers:
                self.lr_schedulers["default"].load_state_dict(state["lr_scheduler"])

    def _build_step_metrics(self, step_result: StepResult) -> Dict[str, Any]:
        metrics = dict(step_result.loss_dict)
        metrics["loss"] = sum(loss.item() for loss in step_result.loss.values())

        finalized_metrics = finalize_metrics(step_result.meta.setdefault("metrics", {}))
        if finalized_metrics:
            metrics["metrics"] = finalized_metrics

        return metrics
