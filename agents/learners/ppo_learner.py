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

                # Forward pass through actor
                actor_loss, kl_div = self._actor_loss(
                    batch_obs, batch_actions, batch_old_log_probs, batch_advantages
                )

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

                critic_loss = self._critic_loss(batch_obs, batch_returns)

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

    def _actor_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes PPO actor loss.

        Returns:
            Tuple of (policy_loss, approx_kl)
        """
        # Get current distribution using learner_inference
        # learner_inference returns raw logits
        batch = {"observations": obs}
        output = self.model.learner_inference(batch)
        logits = output.policies

        # Apply strategy to get distribution from logits
        dist = self.model.policy.strategy.get_distribution(logits)

        log_probs = dist.log_prob(actions)

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(
                ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
            )
            * advantages
        )

        entropy = dist.entropy().mean()
        policy_loss = (
            -torch.min(surr1, surr2).mean() - self.config.entropy_coefficient * entropy
        )

        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean()

        return policy_loss, approx_kl

    def _critic_loss(
        self, observations: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes critic MSE loss.

        Returns:
            Critic loss tensor.
        """
        # Use learner_inference to get raw value logits
        batch = {"observations": observations}
        output = self.model.learner_inference(batch)
        logits = output.values

        # Use the strategy to extract expected scalar values (robust to support-based values)
        values = self.model.value.strategy.to_expected_value(logits)

        # values = values.squeeze(-1) # to_expected_value already handles squeezing if needed
        critic_loss = self.config.critic_coefficient * ((returns - values) ** 2).mean()
        return critic_loss

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
