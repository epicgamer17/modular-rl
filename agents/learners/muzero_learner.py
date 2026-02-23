from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from modules.world_models.inference_output import LearningOutput

import torch
import torch.nn.functional as F
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
        agent_network,
        device,
        num_actions,
        observation_dimensions,
        observation_dtype,
        player_id_mapping: Dict[str, int],
    ):
        super().__init__(
            config=config,
            agent_network=agent_network,
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
            value_prefix=config.value_prefix,
            tau=config.reanalyze_tau,
            multi_process=config.multi_process,
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
        self.loss_pipeline = create_muzero_loss_pipeline(
            config=config,
            device=device,
            agent_network=agent_network,
        )

    @property
    def training_iterations(self) -> int:
        return 1

    def _build_context(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            "has_valid_obs_mask": batch["obs_mask"].to(self.device).bool(),
            "has_valid_action_mask": batch["action_mask"].to(self.device).bool(),
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

    def _predictions_to_dict(self, out: "LearningOutput") -> Dict[str, torch.Tensor]:
        """
        Converts a ``LearningOutput`` NamedTuple into the flat dict expected by
        the MuZero loss pipeline and callbacks.

        The LearningOutput field names are close but not identical to the legacy
        dict keys consumed by ``_run_sequence_pipeline``, so we remap explicitly.

        Args:
            out: LearningOutput produced by ``MuZeroNetwork.learner_inference``.

        Returns:
            Dict mapping pipeline-expected keys to tensors (None values omitted).
        """
        mapping: Dict[str, torch.Tensor] = {
            "values": out.values,
            "policies": out.policies,
            "rewards": out.rewards,
            "to_plays": out.to_plays,
            # loss pipeline uses "latent_states" for consistency loss key lookup
            "latent_states": out.latents,
        }
        # Stochastic MuZero optional fields
        if out.latents_afterstates is not None:
            mapping["latent_afterstates"] = out.latents_afterstates
        if out.chance_logits is not None:
            # loss pipeline uses "chance_codes" for SigmaLoss key lookup
            mapping["chance_codes"] = out.chance_logits
        if out.chance_values is not None:
            mapping["chance_values"] = out.chance_values
        if out.chance_encoder_embeddings is not None:
            # loss pipeline uses "chance_encoder_embeddings" for VQVAECommitmentLoss
            mapping["chance_encoder_embeddings"] = out.chance_encoder_embeddings
        return {k: v for k, v in mapping.items() if v is not None}

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
        predictions = self._predictions_to_dict(learning_out)

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
