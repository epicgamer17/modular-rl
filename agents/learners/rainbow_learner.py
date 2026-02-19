"""
RainbowLearner handles the Q-learning training logic, including buffer management,
optimizer stepping, loss computation, and target network updates.
"""

import time
from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim.sgd import SGD
from torch.optim.adam import Adam

from replay_buffers.buffer_factories import create_dqn_buffer
from replay_buffers.utils import update_per_beta
from losses.losses import LossPipeline, StandardDQNLoss, C51Loss
from modules.utils import get_lr_scheduler


class RainbowLearner:
    """
    RainbowLearner handles the training logic for Rainbow DQN, including
    buffer management, optimizer stepping, and loss computation.
    """

    def __init__(
        self,
        config,
        model: torch.nn.Module,
        target_model: torch.nn.Module,
        device: torch.device,
        num_actions: int,
        observation_dimensions: Tuple[int, ...],
        observation_dtype: torch.dtype,
    ):
        """
        Initializes the RainbowLearner.

        Args:
            config: RainbowConfig with hyperparameters.
            model: The online Q-network.
            target_model: The target Q-network (for stable targets).
            device: Torch device for tensors.
            num_actions: Number of discrete actions.
            observation_dimensions: Shape of observations.
            observation_dtype: Dtype for observations.
        """
        self.config = config
        self.model = model
        self.target_model = target_model
        self.device = device
        self.num_actions = num_actions
        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype
        self.training_step = 0

        # 1. Initialize Replay Buffer
        self.replay_buffer = create_dqn_buffer(
            observation_dimensions=observation_dimensions,
            max_size=config.replay_buffer_size,
            num_actions=num_actions,
            batch_size=config.minibatch_size,
            observation_dtype=observation_dtype,
            config=config,
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

        # 4. Initialize Action Selector for loss calculation (greedy for Double DQN)
        from agents.action_selectors.selectors import ArgmaxSelector

        self.training_selector = ArgmaxSelector()

        # 5. Initialize Loss Pipeline with action selector
        loss_modules = []
        if config.atom_size > 1:
            loss_modules.append(
                C51Loss(config, device, action_selector=self.training_selector)
            )
        else:
            loss_modules.append(
                StandardDQNLoss(config, device, action_selector=self.training_selector)
            )
        self.loss_pipeline = LossPipeline(loss_modules)

        # 5. Create support for distributional RL
        self.support = torch.linspace(
            config.v_min,
            config.v_max,
            config.atom_size,
            device=device,
        )

    def step(self, stats=None) -> Optional[Dict[str, float]]:
        """
        Performs a single training step.

        Args:
            stats: Optional StatTracker for logging metrics.

        Returns:
            Dictionary of loss statistics, or None if buffer is too small.
        """
        if self.replay_buffer.size < self.config.min_replay_buffer_size:
            return None

        start_time = time.time()

        # Run training iterations
        losses = np.zeros(self.config.training_iterations)

        for i in range(self.config.training_iterations):
            # 1. Sample from buffer
            context = self.replay_buffer.sample()

            # 2. Run loss pipeline
            loss, elementwise_loss = self.loss_pipeline.run(self, context)

            # 3. Backpropagation
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if self.config.clipnorm > 0:
                clip_grad_norm_(self.model.parameters(), self.config.clipnorm)

            self.optimizer.step()
            self.lr_scheduler.step()

            # 4. Update priorities (PER)
            self.replay_buffer.update_priorities(
                indices=context["indices"],
                priorities=elementwise_loss.detach(),
                ids=None,
            )

            # 5. Reset noise in NoisyNets
            self.model.reset_noise()
            self.target_model.reset_noise()

            losses[i] = loss.detach().item()

        # Update PER beta
        self.replay_buffer.set_beta(
            update_per_beta(
                self.replay_buffer.beta,
                self.config.per_beta_final,
                self.config.training_steps,
                self.config.per_beta,
            )
        )

        self.training_step += 1

        # Track learner FPS
        if stats is not None:
            duration = time.time() - start_time
            if duration > 0:
                batch_size = (
                    self.config.minibatch_size * self.config.training_iterations
                )
                fps = batch_size / duration
                stats.append("learner_fps", fps)

        # MPS cache clearing
        if self.device.type == "mps" and self.training_step % 100 == 0:
            torch.mps.empty_cache()

        return {"loss": float(losses.mean())}

    def update_target_network(self) -> None:
        """
        Updates the target network weights.
        Uses soft update (EMA) if config.soft_update is True, else hard copy.
        """
        if self.config.soft_update:
            # Soft update: target = beta * target + (1 - beta) * online
            for target_param, online_param in zip(
                self.target_model.parameters(), self.model.parameters()
            ):
                target_param.data.copy_(
                    self.config.ema_beta * target_param.data
                    + (1.0 - self.config.ema_beta) * online_param.data
                )
        else:
            # Hard update
            self.target_model.load_state_dict(self.model.state_dict())

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the online network.

        Args:
            states: Preprocessed observation tensor.

        Returns:
            Q-distribution (logits) or Q-values tensor.
        """
        batch = {"observations": states}
        out = self.model.learner_inference(batch)
        if self.config.atom_size > 1:
            # Distributional RL: Return LOGITS for numerically stable loss calculation (log_softmax)
            return out.q_logits  # (B, actions, atoms) - LOGITS
        return out.q_values  # (B, actions)

    def predict_target(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the target network.

        Args:
            states: Preprocessed observation tensor.

        Returns:
            Q-distribution (logits) or Q-values tensor.
        """
        batch = {"observations": states}
        out = self.target_model.learner_inference(batch)
        if self.config.atom_size > 1:
            # Distributional RL: Return LOGITS for numerically stable loss calculation (log_softmax)
            return out.q_logits  # (B, actions, atoms) - LOGITS
        return out.q_values  # (B, actions)

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

    def select_actions(
        self, q_values: torch.Tensor, info: Optional[list] = None
    ) -> torch.Tensor:
        """
        Selects greedy actions from Q-values with optional legal move masking.
        Used for Double DQN target calculation in the loss pipeline.

        Args:
            q_values: Q-values tensor of shape (B, num_actions) or (B, num_actions, atoms).
            info: Optional list of dicts with 'legal_moves' for action masking.

        Returns:
            Tensor of selected action indices of shape (B,).
        """
        from utils.utils import get_legal_moves, action_mask

        # For distributional RL, reduce atoms to Q-values
        if self.config.atom_size > 1:
            # q_values shape: (B, num_actions, atoms)
            # Expected Q = sum(support * probs) for each action
            q_values = (q_values * self.support).sum(dim=-1)  # (B, num_actions)

        batch_size = q_values.shape[0]

        if info is not None and len(info) > 0 and "legal_moves" in info[0]:
            # Apply action masking for each sample in batch
            actions = []
            for i in range(batch_size):
                legal_moves = info[i].get("legal_moves", None)
                if legal_moves is not None and len(legal_moves) > 0:
                    masked_q = action_mask(
                        q_values[i],
                        legal_moves,
                        mask_value=-float("inf"),
                        device=q_values.device,
                    )
                    actions.append(masked_q.argmax())
                else:
                    actions.append(q_values[i].argmax())
            return torch.stack(actions)

        # No masking - simple greedy
        return q_values.argmax(dim=-1)
