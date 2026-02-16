from typing import Tuple
import torch
from torch import nn
from modules.conv import Conv2dStack
from configs.modules.backbones.conv import ConvConfig


class ConvBackbone(nn.Module):
    """Standard Convolutional backbone implementation using Conv2dStack."""

    def __init__(self, config: ConvConfig, input_shape: Tuple[int, ...]):
        super().__init__()
        self.config = config
        self.input_shape = input_shape

        self.stack = Conv2dStack(
            input_shape=input_shape,
            filters=config.filters,
            kernel_sizes=config.kernel_sizes,
            strides=config.strides,
            activation=config.activation,
            noisy_sigma=config.noisy_sigma,
            norm_type=config.norm_type,
        )

        # Calculate output shape (B, C, H, W)
        # Conv2dStack uses padding='same' (via calculate_same_padding)
        curr_h, curr_w = input_shape[2], input_shape[3]
        for stride in config.strides:
            curr_h = (curr_h + stride - 1) // stride
            curr_w = (curr_w + stride - 1) // stride

        self.output_shape = (input_shape[0], self.stack.output_channels, curr_h, curr_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)

    def initialize(self, initializer: torch.Tensor) -> None:
        self.stack.initialize(initializer)

    def reset_noise(self) -> None:
        self.stack.reset_noise()
