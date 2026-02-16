from typing import Callable, Tuple
from torch import nn, Tensor
from modules.network_block import NetworkBlock
from modules.backbones.factory import BackboneFactory
from modules.heads import ScalarHead
from agent_configs.base_config import Config


class CriticNetwork(nn.Module):
    def __init__(self, config: Config, input_shape: Tuple[int]):
        super().__init__()
        self.config = config

        # 1. Backbone
        self.net = BackboneFactory.create(config.critic_backbone, input_shape)

        # 2. Value Head (Handles scalar vs support automatically)
        input_width = self._get_flat_dim(self.net.output_shape)
        self.head = ScalarHead(config, (self.net.output_shape[0], input_width))

    def _get_flat_dim(self, shape: Tuple[int]) -> int:
        if len(shape) == 4:
            return shape[1] * shape[2] * shape[3]
        return shape[1]

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        self.net.initialize(initializer)
        self.head.initialize(initializer)

    def forward(self, inputs: Tensor):
        x = self.net(inputs)
        x = x.flatten(1, -1)
        return self.head(x)

    def reset_noise(self):
        if hasattr(self.net, "reset_noise"):
            self.net.apply(
                lambda m: m.reset_noise() if hasattr(m, "reset_noise") else None
            )
        self.head.reset_noise()
