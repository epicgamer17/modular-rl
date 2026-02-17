from typing import Tuple
import torch
from torch import nn
from modules.residual import ResidualStack
from configs.modules.backbones.resnet import ResNetConfig


class ResNetBackbone(nn.Module):
    """ResNet backbone implementation using ResidualStack."""

    def __init__(self, config: ResNetConfig, input_shape: Tuple[int, ...]):
        super().__init__()
        self.config = config
        self.input_shape = input_shape

        self.stack = ResidualStack(
            input_shape=input_shape,
            filters=config.filters,
            kernel_sizes=config.kernel_sizes,
            strides=config.strides,
            activation=config.activation,
            noisy_sigma=config.noisy_sigma,
            norm_type=config.norm_type,
        )

        # Calculate output shape (C, H, W)
        # ResidualStack uses padding='same', so spatial size only changes with stride
        curr_h, curr_w = input_shape[1], input_shape[2]
        for stride in config.strides:
            curr_h = (curr_h + stride - 1) // stride
            curr_w = (curr_w + stride - 1) // stride

        self.output_shape = (self.stack.output_channels, curr_h, curr_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)

    def initialize(self, initializer: torch.Tensor) -> None:
        self.stack.initialize(initializer)

    def reset_noise(self) -> None:
        self.stack.reset_noise()
