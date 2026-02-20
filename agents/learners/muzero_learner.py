from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch.optim.adam import Adam
from torch.optim.sgd import SGD

from agents.learners.base import BaseLearner, StepResult
from agents.learners.callbacks import MetricsCallback
from losses.losses import create_muzero_loss_pipeline
from modules.utils import get_lr_scheduler
from replay_buffers.buffer_factories import create_muzero_buffer


class MuZeroLearner(BaseLearner):
    """MuZero learner with shared optimization loop from BaseLearner."""

    def __init__(
        self,
        config,
        model,
        device,
        num_actions,
        observation_dimensions,
        observation_dtype,
    ):
        super().__init__(
            config=config,
            model=model,
            device=device,
            num_actions=num_actions,
            observation_dimensions=observation_dimensions,
            observation_dtype=observation_dtype,
            callbacks=[MetricsCallback()],
        )

        self.replay_buffer = create_muzero_buffer(
            observation_dimensions=observation_dimensions,
            max_size=config.replay_buffer_size,
            num_actions=num_actions,
            num_players=config.game.num_players,
            unroll_steps=config.unroll_steps,
            n_step=config.n_step,
            gamma=config.discount_factor,
            batch_size=config.minibatch_size,
            observation_dtype=observation_dtype,
            alpha=config.per_alpha,
            beta=config.per_beta,
            epsilon=config.per_epsilon,
            use_batch_weights=config.per_use_batch_weights,
            use_initial_max_priority=config.per_use_initial_max_priority,
            lstm_horizon_len=config.lstm_horizon_len,
            value_prefix=config.value_prefix,
            tau=config.reanalyze_tau,
            multi_process=config.multi_process,
        )

        if config.optimizer == Adam:
            self.optimizer = config.optimizer(
                params=model.parameters(),
                lr=config.learning_rate,
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == SGD:
            self.optimizer = config.optimizer(
                params=model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")

        self.lr_scheduler = get_lr_scheduler(self.optimizer, config)
        self.loss_pipeline = create_muzero_loss_pipeline(
            config=config,
            device=device,
            model=model,
            preprocess_fn=self._preprocess_observation,
        )

    @property
    def training_iterations(self) -> int:
        return 1

    def _build_context(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            "has_valid_obs_mask": batch["obs_mask"].to(self.device).bool(),
            "has_valid_action_mask": batch["action_mask"].to(self.device).bool(),
            "target_observations": batch["unroll_observations"].to(self.device),
            "model": self.model,
            "preprocess_fn": self._preprocess_observation,
        }

    def _gradient_scales(self) -> torch.Tensor:
        unroll_steps = self.config.unroll_steps
        scales = [1.0] + [1.0 / unroll_steps] * unroll_steps
        return torch.tensor(scales, device=self.device).reshape(1, -1)

    def _prepare_targets(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        targets = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                targets[key] = value.to(self.device)
        return targets

    def compute_step_result(self, batch: Dict[str, Any], stats=None) -> StepResult:
        batch["observations"] = self._preprocess_observation(batch["observations"])
        targets = self._prepare_targets(batch)
        predictions = self.model.learner_inference(targets)

        weights = targets["weights"].float()
        gradient_scales = self._gradient_scales()
        context = self._build_context(targets)

        loss_mean, loss_dict, priorities = self.loss_pipeline.run(
            predictions=predictions,
            targets=targets,
            context=context,
            weights=weights,
            gradient_scales=gradient_scales,
            config=self.config,
            device=self.device,
        )

        detached_predictions = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in predictions.items()
        }
        detached_targets = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in targets.items()
        }

        return StepResult(
            loss=loss_mean,
            loss_dict=loss_dict,
            priorities=priorities,
            predictions=detached_predictions,
            targets=detached_targets,
        )
