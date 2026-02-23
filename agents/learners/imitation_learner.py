"""
ImitationLearner handles supervised learning for policy imitation,
using CrossEntropy loss to train a network to mimic target policies.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.sgd import SGD

from replay_buffers.buffer_factories import create_nfsp_buffer
from modules.utils import get_lr_scheduler
from agents.learners.base import BaseLearner, StepResult


class ImitationLearner(BaseLearner):
    """
    ImitationLearner trains a policy network via supervised learning.
    It uses CrossEntropy loss to match expert or collected target policies.
    """

    def __init__(
        self,
        config,
        agent_network: nn.Module,
        device: torch.device,
        num_actions: int,
        observation_dimensions: Tuple[int, ...],
        observation_dtype: torch.dtype,
    ):
        """
        Initializes the ImitationLearner.

        Args:
            config: Configuration with hyperparameters (learning_rate, optimizer, etc.).
            agent_network: The policy network to train.
            device: Torch device for tensors.
            num_actions: Number of discrete actions.
            observation_dimensions: Shape of observations.
            observation_dtype: Dtype for observations.
        """
        super().__init__(
            config=config,
            agent_network=agent_network,
            device=device,
            num_actions=num_actions,
            observation_dimensions=observation_dimensions,
            observation_dtype=observation_dtype,
        )

        # 1. Initialize Replay Buffer (Reservoir for SL)
        self.replay_buffer = create_nfsp_buffer(
            observation_dimensions=observation_dimensions,
            observation_dtype=observation_dtype,
            max_size=config.replay_buffer_size,
            num_actions=num_actions,
            batch_size=config.minibatch_size,
        )

        # 2. Initialize Optimizer
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

        # 3. Initialize LR Scheduler
        self.lr_scheduler = get_lr_scheduler(self.optimizer, config)

        # 4. Loss function (CrossEntropy or configurable)
        self.loss_function = getattr(config, "loss_function", nn.CrossEntropyLoss())

    def store(
        self,
        observation: Any,
        legal_moves: list,
        target_policy: torch.Tensor,
    ) -> None:
        """
        Stores a supervised learning transition.

        Args:
            observation: Current observation.
            legal_moves: List of legal action indices.
            target_policy: Target policy distribution to imitate.
        """
        self.replay_buffer.store(
            observations=observation,
            legal_moves=legal_moves,
            target_policies=target_policy,
        )

    def step(self, stats=None) -> Optional[Dict[str, float]]:
        self._policy_total = torch.zeros(self.num_actions, device=self.device)
        self._policy_count = 0
        out = super().step(stats=stats)
        if stats is not None and self._policy_count > 0:
            stats.set("sl_policy", self._policy_total / self._policy_count)
        return out

    def compute_step_result(self, batch, stats=None) -> StepResult:
        observations = batch["observations"].to(self.device)
        targets = batch["target_policies"].to(self.device)

        inf_out = self.agent_network.initial_inference(observations)
        predictions = inf_out.policy.logits

        with torch.inference_mode():
            batch_policy_mean = predictions.detach().mean(dim=0)

        if targets.dim() == 1:
            targets_onehot = torch.zeros(
                targets.shape[0], self.num_actions, device=self.device
            )
            targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
            loss = self.loss_function(predictions, targets_onehot).mean()
        else:
            loss = self.loss_function(predictions, targets).mean()

        return StepResult(
            loss=loss,
            loss_dict={"imitation_loss": float(loss.detach().item())},
            priorities=None,
            meta={"policy_mean": batch_policy_mean},
        )

    def after_optimizer_step(self, batch, step_result: StepResult, stats=None) -> None:
        self.agent_network.reset_noise()
        self._policy_total += step_result.meta["policy_mean"]
        self._policy_count += 1

    def preprocess(self, observation: Any) -> torch.Tensor:
        """
        Preprocesses observation for network input.

        Args:
            observation: Raw observation.

        Returns:
            Preprocessed tensor on the correct device.
        """
        if isinstance(observation, torch.Tensor):
            obs = observation.to(self.device, dtype=torch.float32)
        else:
            obs = torch.tensor(observation, device=self.device, dtype=torch.float32)

        if obs.dim() == len(self.observation_dimensions):
            obs = obs.unsqueeze(0)

        return obs
