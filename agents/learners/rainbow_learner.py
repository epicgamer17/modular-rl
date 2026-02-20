"""
RainbowLearner handles the Q-learning training logic, including buffer management,
optimizer stepping, loss computation, and target network updates.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch.optim.sgd import SGD
from torch.optim.adam import Adam

from replay_buffers.buffer_factories import create_dqn_buffer
from losses.losses import LossPipeline, StandardDQNLoss, C51Loss
from modules.utils import get_lr_scheduler
from agents.learners.base import BaseLearner, StepResult


class RainbowLearner(BaseLearner):
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
        super().__init__(
            config=config,
            model=model,
            device=device,
            num_actions=num_actions,
            observation_dimensions=observation_dimensions,
            observation_dtype=observation_dtype,
        )
        self.target_model = target_model

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

    def compute_step_result(self, batch, stats=None) -> StepResult:
        loss, elementwise_loss = self.loss_pipeline.run(self, batch)
        return StepResult(
            loss=loss,
            loss_dict={"td_loss": float(loss.detach().item())},
            priorities=elementwise_loss.detach(),
        )

    def after_optimizer_step(self, batch, step_result: StepResult, stats=None) -> None:
        self.model.reset_noise()
        self.target_model.reset_noise()

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
        from utils.utils import get_legal_moves

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
                    masked_q = self.training_selector.mask_actions(
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
