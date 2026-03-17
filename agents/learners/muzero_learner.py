from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from modules.world_models.inference_output import LearningOutput

import torch
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim.sgd import SGD

from agents.learners.base import UniversalLearner, StepResult
from agents.learners.batch_iterators import SingleBatchIterator
from agents.learners.callbacks import MetricsCallback
from losses.losses import (
    ChanceQLoss,
    ConsistencyLoss,
    LossPipeline,
    PolicyLoss,
    RewardLoss,
    SigmaLoss,
    ToPlayLoss,
    ValueLoss,
    VQVAECommitmentLoss,
)
from modules.utils import get_lr_scheduler
from replay_buffers.buffer_factories import create_muzero_buffer


class MuZeroLearner(UniversalLearner):
    """MuZero learner with shared optimization loop from UniversalLearner."""

    def __init__(
        self,
        config,
        agent_network,
        device,
        num_actions,
        observation_dimensions,
        observation_dtype,
        player_id_mapping: Dict[str, int],
    ):
        from agents.learners.callbacks import MetricsCallback, ResetNoiseCallback
        super().__init__(
            config=config,
            agent_network=agent_network,
            device=device,
            num_actions=num_actions,
            observation_dimensions=observation_dimensions,
            observation_dtype=observation_dtype,
            callbacks=[MetricsCallback(), ResetNoiseCallback()],
        )

        self.replay_buffer = create_muzero_buffer(
            observation_dimensions=observation_dimensions,
            max_size=config.replay_buffer_size,
            num_actions=num_actions,
            num_players=config.game.num_players,
            player_id_mapping=player_id_mapping,
            unroll_steps=config.unroll_steps,
            n_step=config.n_step,
            gamma=config.discount_factor,
            batch_size=config.minibatch_size,
            observation_dtype=observation_dtype,
            alpha=config.per_alpha,
            beta=config.per_beta_schedule.initial,
            epsilon=config.per_epsilon,
            use_batch_weights=config.per_use_batch_weights,
            use_initial_max_priority=config.per_use_initial_max_priority,
            lstm_horizon_len=config.lstm_horizon_len,
            value_prefix=config.use_value_prefix,
            tau=config.reanalyze_tau,
            multi_process=config.multi_process,
            observation_quantization=config.observation_quantization,
            observation_compression=config.observation_compression,
        )

        if config.optimizer == Adam:
            self.optimizer = config.optimizer(
                params=agent_network.parameters(),
                lr=config.learning_rate,
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == SGD:
            self.optimizer = config.optimizer(
                params=agent_network.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")

        self.lr_scheduler = get_lr_scheduler(self.optimizer, config)

        # 4. Initialize Loss Pipeline
        modules = [
            ValueLoss(config, device),
            PolicyLoss(config, device),
            RewardLoss(config, device),
        ]

        # Optional modules
        if config.game.num_players != 1:
            modules.append(ToPlayLoss(config, device))

        if config.consistency_loss_factor > 0:
            modules.append(ConsistencyLoss(config, device, agent_network))

        if config.stochastic:
            modules.extend(
                [
                    ChanceQLoss(config, device),
                    SigmaLoss(config, device),
                    VQVAECommitmentLoss(config, device),
                ]
            )

        self.loss_pipeline = LossPipeline(modules)

        # 5. Validate Dependencies
        # Verify that our _prepare_targets and _predictions_to_dict provide all required keys
        if len(self.replay_buffer) > 0:
            sample_batch = self.replay_buffer.sample()
            dummy_targets = self._prepare_targets(sample_batch)
            if self.config.consistency_loss_factor > 0:
                dummy_targets["consistency_targets"] = torch.zeros((1, 1, 1))

            # We assume learner_inference provides these keys based on _predictions_to_dict
            dummy_pred_keys = {
                "values",
                "policies",
                "rewards",
                "to_plays",
                "latents",
            }
            if config.stochastic:
                dummy_pred_keys.update(
                    {"chance_codes", "chance_values", "chance_encoder_embeddings"}
                )

            self.loss_pipeline.validate_dependencies(
                network_output_keys=dummy_pred_keys,
                target_keys=set(dummy_targets.keys()),
            )

        from agents.learners.callbacks import PriorityUpdaterCallback
        self.callbacks.callbacks.append(PriorityUpdaterCallback(self.replay_buffer))

        # PER beta schedule
        from utils.schedule import create_schedule

        self.schedules: Dict[str, Any] = {}
        if (
            hasattr(self.config, "per_beta_schedule")
            and self.config.per_beta_schedule is not None
        ):
            self.schedules["per_beta"] = create_schedule(self.config.per_beta_schedule)

    def step(self, stats=None) -> Any:
        """Bridges the old trainer call pattern: checks buffer, builds iterator, delegates."""
        if self.replay_buffer.size < self.config.min_replay_buffer_size:
            return None

        iterator = SingleBatchIterator(self.replay_buffer)
        result = super().step(batch_iterator=iterator, stats=stats)

        # Step PER beta schedule
        for name, schedule in self.schedules.items():
            schedule.step()
            if name == "per_beta":
                self.replay_buffer.set_beta(schedule.get_value())

        return result

    def _build_context(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            "has_valid_obs_mask": batch["has_valid_obs_mask"].to(self.device).bool(),
            "has_valid_action_mask": batch["has_valid_action_mask"]
            .to(self.device)
            .bool(),
            "is_same_game": batch["is_same_game"].to(self.device).bool(),
        }

    def _prepare_consistency_targets(
        self, unroll_observations: torch.Tensor
    ) -> torch.Tensor:
        """Build detached target embeddings for EfficientZero consistency loss."""
        real_obs = self._preprocess_observation(unroll_observations)
        batch_size, unroll_len = real_obs.shape[:2]
        flat_obs = real_obs.flatten(0, 1)

        initial_out = self.agent_network.obs_inference(flat_obs)
        real_latents = initial_out.network_state.dynamics
        # Clone to promote from inference_mode tensors (created by obs_inference's
        # @torch.inference_mode() decorator) to normal autograd-tracked tensors.
        # Without this, project() cannot save the tensor for backward.
        real_latents = real_latents.clone()

        proj_targets = self.agent_network.project(real_latents, grad=False)
        normalized_targets = F.normalize(proj_targets, p=2.0, dim=-1, eps=1e-5)
        return normalized_targets.reshape(batch_size, unroll_len, -1).detach()

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
        """
        Runs one forward/loss pass for a sampled MuZero batch.

        Args:
            batch: Dict produced by the MuZero replay buffer sampler.
            stats: Optional stat tracker.

        Returns:
            StepResult with scalar loss, loss breakdown dict, and PER priorities.
        """
        batch["observations"] = self._preprocess_observation(batch["observations"])
        targets = self._prepare_targets(batch)
        if self.config.consistency_loss_factor > 0:
            targets["consistency_targets"] = self._prepare_consistency_targets(
                targets["unroll_observations"]
            )
        else:
            targets["consistency_targets"] = None
        learning_out = self.agent_network.learner_inference(targets)

        # Convert the typed NamedTuple to the dict the loss pipeline and
        # callbacks expect.
        weights = targets["weights"].float()
        gradient_scales = self._gradient_scales()
        context = self._build_context(targets)

        loss_mean, loss_dict, priorities = self.loss_pipeline.run(
            predictions=learning_out,
            targets=targets,
            context=context,
            weights=weights,
            gradient_scales=gradient_scales,
        )

        detached_predictions = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in learning_out._asdict().items()
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
