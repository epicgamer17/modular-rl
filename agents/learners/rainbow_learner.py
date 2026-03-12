"""
RainbowLearner handles the Q-learning training logic, including buffer management,
optimizer stepping, loss computation, and target network updates.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch.optim.sgd import SGD
from torch.optim.adam import Adam

from replay_buffers.buffer_factories import create_dqn_buffer
from losses.basic_losses import C51LossModule, StandardDQNLossModule
from modules.utils import get_lr_scheduler
from utils.schedule import create_schedule
from agents.learners.base import UniversalLearner, StepResult
from agents.learners.target_builders import DQNTargetBuilder
from modules.world_models.inference_output import LearningOutput


class RainbowLearner(UniversalLearner):
    """
    RainbowLearner handles the training logic for Rainbow DQN, including
    buffer management, optimizer stepping, and loss computation.
    """

    def __init__(
        self,
        config,
        agent_network: torch.nn.Module,
        target_agent_network: torch.nn.Module,
        device: torch.device,
        num_actions: int,
        observation_dimensions: Tuple[int, ...],
        observation_dtype: torch.dtype,
    ):
        """
        Initializes the RainbowLearner.

        Args:
            config: RainbowConfig with hyperparameters.
            agent_network: The online Q-network.
            target_agent_network: The target Q-network (for stable targets).
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
        self.target_agent_network = target_agent_network
        # Link target network to online network for unrolled evaluations in learner_inference
        self.agent_network.target_network = target_agent_network

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

        # 4. Initialize Action Selector for loss calculation (greedy for Double DQN)
        from agents.action_selectors.selectors import ArgmaxSelector

        self.training_selector = ArgmaxSelector()

        # 5. Initialize architecture-agnostic TD loss module
        from losses.losses import LossPipeline

        if config.atom_size > 1:
            self.td_loss_module = C51LossModule(
                config=config,
                device=device,
                action_selector=self.training_selector,
            )
        else:
            self.td_loss_module = StandardDQNLossModule(
                config=config,
                device=device,
                action_selector=self.training_selector,
            )
        self.loss_pipeline = LossPipeline([self.td_loss_module])

        # 6. Initialize Target Builder
        self.target_builder = DQNTargetBuilder(config, device)
        if config.atom_size > 1:
            self.loss_pipeline.validate_dependencies(
                network_output_keys={"q_logits", "next_q_logits"},
                target_keys={"target_next_q_logits", "actions", "rewards", "dones"},
            )
        else:
            self.loss_pipeline.validate_dependencies(
                network_output_keys={"q_values", "next_q_values"},
                target_keys={"target_next_q_values", "actions", "rewards", "dones"},
            )

        # 5. Create support for distributional RL
        self.support = torch.linspace(
            config.v_min,
            config.v_max,
            config.atom_size,
            device=device,
        )

    def _init_schedules(self):
        super()._init_schedules()
        self.schedules["epsilon"] = create_schedule(self.config.epsilon_schedule)

    @property
    def current_epsilon(self) -> float:
        if "epsilon" in self.schedules:
            return self.schedules["epsilon"].get_value()
        return 0.0

    def after_optimizer_step(self, batch, step_result: StepResult, stats=None) -> None:
        self.agent_network.reset_noise()
        self.target_agent_network.reset_noise()

    def update_target_network(self) -> None:
        """
        Updates the target network weights.
        Uses soft update (EMA) if config.soft_update is True, else hard copy.
        """
        from modules.utils import get_clean_state_dict

        with torch.no_grad():
            clean_state = get_clean_state_dict(self.agent_network)
            if self.config.soft_update:
                target_state = self.target_agent_network.state_dict()
                for k, v in clean_state.items():
                    if k in target_state:
                        # Soft update: target = beta * target + (1 - beta) * online
                        # Ensure we only update floating point tensors (like weights)
                        if target_state[k].is_floating_point():
                            target_state[k].data.mul_(self.config.ema_beta).add_(
                                v.data, alpha=1.0 - self.config.ema_beta
                            )
                        else:
                            target_state[k].data.copy_(v.data)
            else:
                # Hard update
                self.target_agent_network.load_state_dict(clean_state, strict=False)

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the online network.

        Args:
            states: Preprocessed observation tensor.

        Returns:
            Q-distribution (logits) or Q-values tensor.
        """
        batch = {"observations": states}
        out = self.agent_network.learner_inference(batch)
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
        out = self.target_agent_network.learner_inference(batch)
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

    def save_checkpoint(self, path: str):
        """
        Saves Rainbow learner state (online/target weights, optimizer, step).
        """
        checkpoint = {
            "agent_network": self.agent_network.state_dict(),
            "target_agent_network": self.target_agent_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": (
                self.lr_scheduler.state_dict() if self.lr_scheduler else None
            ),
            "training_step": self.training_step,
            "schedules": {k: v.state_dict() for k, v in self.schedules.items()},
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Loads Rainbow learner state from path.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.agent_network.load_state_dict(checkpoint["agent_network"])
        self.target_agent_network.load_state_dict(checkpoint["target_agent_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("lr_scheduler") and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if "schedules" in checkpoint:
            for k, v in checkpoint["schedules"].items():
                if k in self.schedules:
                    self.schedules[k].load_state_dict(v)
        self.training_step = checkpoint.get("training_step", 0)

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
