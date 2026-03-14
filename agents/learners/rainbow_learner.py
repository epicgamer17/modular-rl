"""
RainbowLearner handles the Q-learning training logic, including buffer management,
optimizer stepping, loss computation, and target network updates.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch.optim.sgd import SGD
from torch.optim.adam import Adam

from modules.agent_nets.modular import ModularAgentNetwork
from replay_buffers.buffer_factories import create_dqn_buffer
from losses.losses import C51Loss, StandardDQNLoss
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
        agent_network: ModularAgentNetwork,
        target_agent_network: ModularAgentNetwork,
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
            self.td_loss_module = C51Loss(
                config=config,
                device=device,
                action_selector=self.training_selector,
            )
        else:
            self.td_loss_module = StandardDQNLoss(
                config=config,
                device=device,
                action_selector=self.training_selector,
            )
        self.loss_pipeline = LossPipeline([self.td_loss_module])

        # 6. Initialize Target Builder
        self.target_builder = DQNTargetBuilder(
            config, device, self.target_agent_network
        )
        if config.atom_size > 1:
            self.loss_pipeline.validate_dependencies(
                network_output_keys={"q_logits"},
                target_keys={"q_logits", "actions"},
            )
        else:
            self.loss_pipeline.validate_dependencies(
                network_output_keys={"q_values"},
                target_keys={"q_values", "actions"},
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
            # Resample target noise ONLY after a weight sync to maintain stable C51 targets

            if hasattr(self.target_agent_network, "reset_noise") and callable(
                self.target_agent_network.reset_noise
            ):
                self.target_agent_network.reset_noise()

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
