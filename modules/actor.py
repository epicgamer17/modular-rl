from typing import Callable, Tuple
from torch import nn, Tensor
from modules.network_block import NetworkBlock
from modules.backbones.factory import BackboneFactory
from modules.heads import (
    CategoricalHead,
    ContinuousHead,
)
from agent_configs.base_config import Config
from modules.utils import NetworkOutput


class ActorNetwork(nn.Module):
    def __init__(
        self,
        config: Config,
        input_shape: Tuple[int],
        output_size: int,
    ):
        super().__init__()
        self.config = config

        # 1. Backbone
        self.net = BackboneFactory.create(config.actor_backbone, input_shape)

        # 2. Output Head
        # Note: NetworkBlock calculates the correct flattened output width automatically
        input_width = self._get_flat_dim(self.net.output_shape)

        if self.config.game.is_discrete:
            self.head = CategoricalHead(
                config, (self.net.output_shape[0], input_width), output_size
            )
        else:
            self.head = ContinuousHead(
                config, (self.net.output_shape[0], input_width), output_size
            )

    def _get_flat_dim(self, shape: Tuple[int]) -> int:
        if len(shape) == 4:  # (B, C, H, W)
            return shape[1] * shape[2] * shape[3]
        elif len(shape) == 2:  # (B, D)
            return shape[1]
        raise ValueError(f"Unknown shape {shape}")

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        self.net.initialize(initializer)
        self.head.initialize(initializer)

    def forward(self, inputs: Tensor):
        # Backbone
        x = self.net(inputs)

        # Flatten for the head if necessary
        x = x.flatten(1, -1)

        # Head
        return self.head(x)

    def reset_noise(self):
        # NetworkBlock handles recursion for reset_noise if layers support it
        if hasattr(self.net, "reset_noise"):
            self.net.apply(
                lambda m: m.reset_noise() if hasattr(m, "reset_noise") else None
            )
        self.head.reset_noise()
