"""
ImitationLearner handles supervised learning for policy imitation,
using CrossEntropy loss to train a network to mimic target policies.
"""

import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim.adam import Adam
from torch.optim.sgd import SGD

from replay_buffers.buffer_factories import create_nfsp_buffer
from modules.utils import get_lr_scheduler


class ImitationLearner:
    """
    ImitationLearner trains a policy network via supervised learning.
    It uses CrossEntropy loss to match expert or collected target policies.
    """

    def __init__(
        self,
        config,
        model: nn.Module,
        device: torch.device,
        num_actions: int,
        observation_dimensions: Tuple[int, ...],
        observation_dtype: torch.dtype,
    ):
        """
        Initializes the ImitationLearner.

        Args:
            config: Configuration with hyperparameters (learning_rate, optimizer, etc.).
            model: The policy network to train.
            device: Torch device for tensors.
            num_actions: Number of discrete actions.
            observation_dimensions: Shape of observations.
            observation_dtype: Dtype for observations.
        """
        self.config = config
        self.model = model
        self.device = device
        self.num_actions = num_actions
        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype
        self.training_step = 0

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

        # 3. Initialize LR Scheduler
        self.lr_scheduler = get_lr_scheduler(self.optimizer, config)

        # 4. Loss function (CrossEntropy or configurable)
        self.loss_function = getattr(config, "loss_function", nn.CrossEntropyLoss())

    def store(
        self,
        observation: Any,
        info: Dict[str, Any],
        target_policy: torch.Tensor,
    ) -> None:
        """
        Stores a supervised learning transition.

        Args:
            observation: Current observation.
            info: Current info dict (for legal moves mask).
            target_policy: Target policy distribution to imitate.
        """
        self.replay_buffer.store(
            observations=observation,
            infos=info,
            target_policies=target_policy,
        )

    def step(self, stats=None) -> Optional[Dict[str, float]]:
        """
        Performs supervised learning training steps.

        Args:
            stats: Optional StatTracker for logging metrics.

        Returns:
            Dictionary of loss statistics, or None if buffer is too small.
        """
        min_size = getattr(
            self.config, "min_replay_buffer_size", self.config.minibatch_size
        )
        if self.replay_buffer.size < min_size:
            return None

        start_time = time.time()
        training_iterations = getattr(self.config, "training_iterations", 1)
        losses = np.zeros(training_iterations)
        total_policy = torch.zeros(self.num_actions, device=self.device)

        for i in range(training_iterations):
            # 1. Sample from buffer
            sample = self.replay_buffer.sample()
            observations = sample["observations"].to(self.device)
            targets = sample["target_policies"].to(self.device)

            # 2. Forward pass
            # initial_inference returns Categorical policy.
            # We need logits for CrossEntropyLoss usually, or we can use NLLLoss with log_probs.
            # If we used Categorical(logits=...), .logits attribute exists.
            # If we used Categorical(probs=...), .logits is computed.
            inf_out = self.model.initial_inference(observations)
            predictions = inf_out.policy.logits

            # Accumulate average policy for plotting
            with torch.inference_mode():
                total_policy += predictions.detach().mean(dim=0)

            # 3. Compute loss
            # CategoricalCrossentropyLoss expects both predicted and target as distributions
            # If targets are one-hot, pass them directly; if indices, we need to one-hot them
            if targets.dim() == 1:
                # targets are indices - convert to one-hot
                targets_onehot = torch.zeros(
                    targets.shape[0], self.num_actions, device=self.device
                )
                targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
                loss = self.loss_function(predictions, targets_onehot).mean()
            else:
                # targets are already distributions (one-hot or soft)
                loss = self.loss_function(predictions, targets).mean()

            # 4. Backpropagation
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            clipnorm = getattr(self.config, "clipnorm", 0)
            if clipnorm > 0:
                clip_grad_norm_(self.model.parameters(), clipnorm)

            self.optimizer.step()
            self.lr_scheduler.step()

            # 5. Reset noise if applicable
            self.model.reset_noise()

            losses[i] = loss.detach().item()

        self.training_step += 1

        # Track stats
        if stats is not None:
            # 1. Learner FPS
            duration = time.time() - start_time
            if duration > 0:
                batch_size = self.config.minibatch_size * training_iterations
                fps = batch_size / duration
                stats.append("learner_fps", fps)

            # 2. SL Policy distribution
            avg_policy = total_policy / training_iterations
            stats.set("sl_policy", avg_policy)

        # MPS cache clearing
        if self.device.type == "mps" and self.training_step % 100 == 0:
            torch.mps.empty_cache()

        return {"loss": float(losses.mean())}

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
