from typing import Callable, Tuple
from agent_configs.ppo_config import PPOConfig
from torch import Tensor
from modules.actor import ActorNetwork
from modules.critic import CriticNetwork
import torch.nn as nn


class PPONetwork(nn.Module):
    # This combines the two new base modules
    def __init__(
        self, config: PPOConfig, input_shape: Tuple[int], output_size: int, discrete
    ):
        super().__init__()
        # TODO: ADD SHARED LAYERS
        self.critic = CriticNetwork(config, input_shape)
        self.actor = ActorNetwork(config, input_shape, output_size)

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        self.actor.initialize(initializer)
        self.critic.initialize(initializer)

    def forward(self, inputs: Tensor):
        return self.actor(inputs), self.critic(inputs)
