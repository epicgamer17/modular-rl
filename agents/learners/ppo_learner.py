"""
PPOLearner handles the policy gradient training logic, including buffer management,
optimizer stepping, loss computation with clipped surrogate objective, and GAE calculation.
"""

import time
from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim.sgd import SGD
from torch.optim.adam import Adam

from losses.basic_losses import PPOPolicyLoss, PPOValueLoss
from modules.utils import get_lr_scheduler
from replay_buffers.buffer_factories import create_ppo_buffer
from agents.learners.base import BaseLearner, StepResult


class PPOLearner(BaseLearner):
    """
    PPOLearner handles the training logic for PPO, including buffer management,
    optimizer stepping, and loss computation with clipped surrogate objective.
    """

    def __init__(
        self,
        config,
        model: torch.nn.Module,
        device: torch.device,
        num_actions: int,
        observation_dimensions: Tuple[int, ...],
        observation_dtype: torch.dtype,
    ):
        """
        Initializes the PPOLearner.

        Args:
            config: PPOConfig with hyperparameters.
            model: The policy-value network (shared or separate actor/critic).
            device: Torch device for tensors.
            num_actions: Number of discrete actions.
            observation_dimensions: Shape of observations.
            observation_dtype: Dtype for observations.
        """
        super().__init__(
            config=config,
            model=model,
            device=device,
            num_actions=num_actions,
            observation_dimensions=observation_dimensions,
            observation_dtype=observation_dtype,
        )
        self.discrete_action_space = (
            True  # PPO supports continuous too, but we focus on discrete
        )

        # 1. Initialize On-Policy Replay Buffer (stores one epoch of rollouts)
        self.replay_buffer = create_ppo_buffer(
            observation_dimensions=observation_dimensions,
            max_size=config.replay_buffer_size,
            gamma=config.discount_factor,
            gae_lambda=config.gae_lambda,
            num_actions=num_actions,
            observation_dtype=observation_dtype,
        )

        # 2. Initialize Optimizers (separate for actor and critic)
        self.policy_optimizer = self._create_optimizer(
            self.model.policy.parameters(),
            config.actor,
        )
        self.value_optimizer = self._create_optimizer(
            self.model.value.parameters(),
            config.critic,
        )

        # 3. Initialize LR Schedulers
        self.policy_scheduler = get_lr_scheduler(self.policy_optimizer, config)
        self.value_scheduler = get_lr_scheduler(self.value_optimizer, config)
        self.policy_loss_module = PPOPolicyLoss(
            clip_param=self.config.clip_param,
            entropy_coefficient=self.config.entropy_coefficient,
            policy_strategy=getattr(self.model.policy, "strategy", None),
        )
        self.value_loss_module = PPOValueLoss(
            critic_coefficient=self.config.critic_coefficient,
            atom_size=getattr(self.config, "atom_size", 1),
            v_min=getattr(self.config, "v_min", None),
            v_max=getattr(self.config, "v_max", None),
            value_strategy=getattr(self.model.value, "strategy", None),
        )

    def _create_optimizer(self, params, sub_config) -> torch.optim.Optimizer:
        """Creates optimizer based on config."""
        if sub_config.optimizer == Adam:
            return Adam(
                params=params,
                lr=self.config.learning_rate,
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif sub_config.optimizer == SGD:
            return SGD(
                params=params,
                lr=self.config.learning_rate,
                momentum=getattr(self.config, "momentum", 0.0),
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {sub_config.optimizer}")

    def step(self, stats=None, step=None) -> Optional[Dict[str, float]]:
        """
        Performs PPO training iterations over collected data.

        Args:
            stats: Optional StatTracker for logging metrics.
            step: Optional global training step (used for scheduling).

        Returns:
            Dictionary of loss statistics.
        """
        if self.replay_buffer.size < self.replay_buffer.batch_size:
            return None

        start_time = time.time()

        # Get current rollout batch from modular replay buffer.
        batch = self.replay_buffer.sample()
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["log_probabilities"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)

        # Training indices for minibatch sampling
        num_samples = observations.shape[0]
        indices = torch.randperm(num_samples, device=self.device)
        minibatch_size = max(1, num_samples // self.config.num_minibatches)

        actor_losses = []
        critic_losses = []
        kl_divergences = []

        # Actor training iterations
        early_stop = False
        for iteration in range(self.config.train_policy_iterations):
            if early_stop:
                break

            for start in range(0, num_samples, minibatch_size):
                end = start + minibatch_size
                batch_indices = indices[start:end]

                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]

                loss_terms = self.compute_loss(
                    batch={"observations": batch_obs},
                    actions=batch_actions,
                    old_log_probs=batch_old_log_probs,
                    advantages=batch_advantages,
                )
                actor_loss = loss_terms["policy_loss"]
                kl_div = loss_terms["approx_kl"]

                self.policy_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()

                if self.config.actor.clipnorm > 0:
                    clip_grad_norm_(
                        self.model.policy.parameters(), self.config.actor.clipnorm
                    )

                self.policy_optimizer.step()

                actor_losses.append(actor_loss.detach().item())
                kl_divergences.append(kl_div.detach().item())

            # KL divergence early stopping
            mean_kl = (
                np.mean(kl_divergences[-self.config.num_minibatches :])
                if kl_divergences
                else 0
            )
            if mean_kl > 1.5 * self.config.target_kl:
                early_stop = True

        # Critic training iterations
        for iteration in range(self.config.train_value_iterations):
            for start in range(0, num_samples, minibatch_size):
                end = start + minibatch_size
                batch_indices = indices[start:end]

                batch_obs = observations[batch_indices]
                batch_returns = returns[batch_indices]

                loss_terms = self.compute_loss(
                    batch={"observations": batch_obs},
                    returns=batch_returns,
                )
                critic_loss = loss_terms["value_loss"]

                self.value_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()

                if self.config.critic.clipnorm > 0:
                    clip_grad_norm_(
                        self.model.value.parameters(), self.config.critic.clipnorm
                    )

                self.value_optimizer.step()

                critic_losses.append(critic_loss.detach().item())

        # Step schedulers
        self.policy_scheduler.step()
        self.value_scheduler.step()

        # Clear buffer for next epoch
        self.replay_buffer.clear()

        self.training_step += 1

        # Track learner FPS
        if stats is not None:
            duration = time.time() - start_time
            if duration > 0:
                fps = num_samples / duration
                stats.append("learner_fps", fps)

        # MPS cache clearing
        if self.device.type == "mps" and self.training_step % 100 == 0:
            torch.mps.empty_cache()

        # track loss metrics
        return {
            "policy_loss": np.mean(actor_losses) if actor_losses else 0,
            "value_loss": np.mean(critic_losses) if critic_losses else 0,
            "kl_divergence": np.mean(kl_divergences) if kl_divergences else 0,
        }

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        old_log_probs: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes PPO losses from raw learner logits.
        """
        unroll_out = self.model.learner_inference(batch)
        losses: Dict[str, torch.Tensor] = {}

        if (
            actions is not None
            and old_log_probs is not None
            and advantages is not None
            and unroll_out.policies is not None
        ):
            policy_loss, approx_kl = self.policy_loss_module.compute(
                policy_logits=unroll_out.policies,
                actions=actions,
                old_log_probs=old_log_probs,
                advantages=advantages,
            )
            losses["policy_loss"] = policy_loss
            losses["approx_kl"] = approx_kl

        if returns is not None and unroll_out.values is not None:
            losses["value_loss"] = self.value_loss_module.compute(
                value_logits=unroll_out.values,
                returns=returns,
            )

        return losses

    def preprocess(self, observation: Any) -> torch.Tensor:
        """
        Preprocesses observation for network input.

        Args:
            observation: Raw observation.

        Returns:
            Preprocessed tensor on the correct device.
        """
        return self._preprocess_observation(observation)

    @property
    def size(self) -> int:
        """Returns current number of stored transitions."""
        return self.replay_buffer.size

    # PPO keeps a custom optimization loop (dual optimizers, KL early stopping).
    def compute_step_result(self, batch: Dict[str, Any], stats=None) -> StepResult:
        raise NotImplementedError("PPOLearner uses a custom step implementation.")
