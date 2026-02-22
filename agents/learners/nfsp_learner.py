"""
NFSPLearner handles the training logic for NFSP, coordinating updates for both
the Best Response (RL) network and the Average Strategy (SL) network.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from agents.learners.rainbow_learner import RainbowLearner
from agents.learners.imitation_learner import ImitationLearner


class NFSPLearner:
    """
    NFSPLearner manages the dual-learning process of NFSP.
    It composes a RainbowLearner for Best Response (RL) and an
    ImitationLearner for Average Strategy (SL).
    """

    def __init__(
        self,
        config,
        best_response_agent_network: nn.Module,
        best_response_target_agent_network: nn.Module,
        average_agent_network: nn.Module,
        device: torch.device,
        num_actions: int,
        observation_dimensions: Tuple[int, ...],
        observation_dtype: torch.dtype,
    ):
        """
        Initializes the NFSPLearner.

        Args:
            config: NFSPConfig with hyperparameters.
            best_response_agent_network: Network for Best Response.
            best_response_target_agent_network: Target network for Best Response.
            average_agent_network: Network for Average Strategy.
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
        self.rl_learner = RainbowLearner(
            config=config.rl_configs[0],
            agent_network=best_response_agent_network,
            target_agent_network=best_response_target_agent_network,
            device=device,
            num_actions=num_actions,
            observation_dimensions=observation_dimensions,
            observation_dtype=observation_dtype,
        )

        # 2. Initialize Average Strategy (SL) Learner via composition
        self.sl_learner = ImitationLearner(
            config=config.sl_configs[0],
            agent_network=average_agent_network,
            device=device,
            num_actions=num_actions,
            observation_dimensions=observation_dimensions,
            observation_dtype=observation_dtype,
        )

    @property
    def sl_optimizer(self) -> torch.optim.Optimizer:
        """Backwards compatibility: expose SL optimizer from composed learner."""
        return self.sl_learner.optimizer

    @property
    def sl_replay_buffer(self):
        """Backwards compatibility: expose SL buffer from composed learner."""
        return self.sl_learner.replay_buffer

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
            terminated=done,
            truncated=False,
            dones=done,
        )

        # If best_response was used, store in SL reservoir buffer
        if policy_used == "best_response":
            target_policy = torch.zeros(self.num_actions)
            target_policy[action] = 1.0
            self.sl_learner.store(
                observation=observation,
                info=info,
                target_policy=target_policy,
            )

    def step(self, stats=None) -> Optional[Dict[str, float]]:
        """
        Performs training steps for both RL and SL components.

        Args:
            stats: Optional StatTracker for logging metrics.

        Returns:
            Dictionary of combined loss statistics.
        """
        metrics = {}

        # 1. RL Step
        rl_metrics = self.rl_learner.step(stats)
        if rl_metrics:
            metrics.update({f"rl_{k}": v for k, v in rl_metrics.items()})

        # 2. SL Step (delegated to ImitationLearner)
        sl_metrics = self.sl_learner.step(stats)
        if sl_metrics:
            metrics.update({f"sl_{k}": v for k, v in sl_metrics.items()})

        self.training_step += 1
        return metrics if metrics else None

    def update_target_network(self) -> None:
        """Updates the RL target network."""
        self.rl_learner.update_target_network()

    def preprocess(self, observation: Any) -> torch.Tensor:
        """Delegates preprocessing to RL learner."""
        return self.rl_learner.preprocess(observation)
