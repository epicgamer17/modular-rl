"""
NFSPLearner handles the training logic for NFSP, coordinating updates for both
the Best Response (RL) network and the Average Strategy (SL) network.
"""

import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim.adam import Adam
from torch.optim.sgd import SGD

from agents.learners.rainbow_learner import RainbowLearner
from replay_buffers.buffer_factories import create_nfsp_buffer
from modules.utils import get_lr_scheduler


class NFSPLearner:
    """
    NFSPLearner manages the dual-learning process of NFSP.
    It contains a Best Response learner (RL) and an Average Strategy learner (SL).
    """

    def __init__(
        self,
        config,
        best_response_model: nn.Module,
        best_response_target_model: nn.Module,
        average_model: nn.Module,
        device: torch.device,
        num_actions: int,
        observation_dimensions: Tuple[int, ...],
        observation_dtype: torch.dtype,
    ):
        """
        Initializes the NFSPLearner.

        Args:
            config: NFSPConfig with hyperparameters.
            best_response_model: Network for Best Response.
            best_response_target_model: Target network for Best Response.
            average_model: Network for Average Strategy.
            device: Torch device for tensors.
            num_actions: Number of discrete actions.
            observation_dimensions: Shape of observations.
            observation_dtype: Dtype for observations.
        """
        self.config = config
        self.device = device
        self.num_actions = num_actions
        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype
        self.training_step = 0

        # 1. Initialize Best Response (RL) Learner
        # We wrap RainbowLearner for RL updates
        self.rl_learner = RainbowLearner(
            config=config.rl_configs[0],
            model=best_response_model,
            target_model=best_response_target_model,
            device=device,
            num_actions=num_actions,
            observation_dimensions=observation_dimensions,
            observation_dtype=observation_dtype,
        )

        # 2. Initialize Average Strategy (SL) components
        self.average_model = average_model
        sl_config = config.sl_configs[0]

        # SL Replay Buffer (Reservoir)
        self.sl_replay_buffer = create_nfsp_buffer(
            observation_dimensions=observation_dimensions,
            observation_dtype=observation_dtype,
            max_size=sl_config.replay_buffer_size,
            num_actions=num_actions,
            batch_size=sl_config.minibatch_size,
        )

        # SL Optimizer
        if sl_config.optimizer == Adam:
            self.sl_optimizer = sl_config.optimizer(
                params=average_model.parameters(),
                lr=sl_config.learning_rate,
                eps=sl_config.adam_epsilon,
                weight_decay=sl_config.weight_decay,
            )
        elif sl_config.optimizer == SGD:
            self.sl_optimizer = sl_config.optimizer(
                params=average_model.parameters(),
                lr=sl_config.learning_rate,
                momentum=sl_config.momentum,
                weight_decay=sl_config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported SL optimizer: {sl_config.optimizer}")

        self.sl_lr_scheduler = get_lr_scheduler(self.sl_optimizer, sl_config)

    def store(
        self,
        observation: Any,
        info: Dict[str, Any],
        action: int,
        reward: float,
        next_observation: Any,
        next_info: Dict[str, Any],
        done: bool,
        policy_used: str,
    ) -> None:
        """
        Stores a transition in the appropriate replay buffers.

        Args:
            observation: Current observation.
            info: Current info.
            action: Action taken.
            reward: Reward received.
            next_observation: Next observation.
            next_info: Next info.
            done: Whether the episode finished.
            policy_used: Either "best_response" or "average_strategy".
        """
        # Always store in RL replay buffer
        self.rl_learner.replay_buffer.store(
            observations=observation,
            infos=info,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            next_infos=next_info,
            dones=done,
        )

        # If best_response was used, store in SL reservoir buffer
        if policy_used == "best_response":
            # NFSP stores (s, a) as a supervised target
            # Convert action to a one-hot or target distribution if needed,
            # but NFSPReservoirBuffer.store seems to handle action directly?
            # Let's check NFSPReservoirBuffer.store signature (assuming it takes state, info, target_policy)
            target_policy = torch.zeros(self.num_actions)
            target_policy[action] = 1.0
            self.sl_replay_buffer.store(
                observations=observation, infos=info, target_policies=target_policy
            )

    def step(self, stats=None) -> Optional[Dict[str, float]]:
        """
        Performs training steps for both RL and SL components.
        """
        metrics = {}

        # 1. RL Step
        rl_metrics = self.rl_learner.step(stats)
        if rl_metrics:
            metrics.update({f"rl_{k}": v for k, v in rl_metrics.items()})

        # 2. SL Step
        sl_metrics = self._sl_step()
        if sl_metrics:
            metrics.update({f"sl_{k}": v for k, v in sl_metrics.items()})

        self.training_step += 1
        return metrics if metrics else None

    def _sl_step(self) -> Optional[Dict[str, float]]:
        """
        Performs supervised learning update for the Average Strategy network.
        """
        sl_config = self.config.sl_configs[0]
        if self.sl_replay_buffer.size < sl_config.min_replay_buffer_size:
            return None

        losses = []
        for _ in range(sl_config.training_iterations):
            sample = self.sl_replay_buffer.sample()
            observations = sample["observations"].to(self.device)
            targets = sample["targets"].to(self.device)

            # In NFSP, we often apply action masking even during SL training if applicable
            # But for simplicity, let's start with raw output
            predictions = self.average_model(observations)

            # Loss function (Cross Entropy / Policy Imitation)
            # NFSP typically uses log-likelihood: L = -E[log Pi(a|s)]
            # If using CategoricalHead, predictions are already probabilities or logits
            # sl_config.loss_function usually handles this
            loss = sl_config.loss_function(predictions, targets).mean()

            self.sl_optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if sl_config.clipnorm > 0:
                clip_grad_norm_(self.average_model.parameters(), sl_config.clipnorm)

            self.sl_optimizer.step()
            self.sl_lr_scheduler.step()

            losses.append(loss.detach().item())

        return {"loss": float(np.mean(losses))}

    def update_target_network(self) -> None:
        """Updates the RL target network."""
        self.rl_learner.update_target_network()

    def preprocess(self, observation: Any) -> torch.Tensor:
        """Delegates preprocessing to RL learner."""
        return self.rl_learner.preprocess(observation)
