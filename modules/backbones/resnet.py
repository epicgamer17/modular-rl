from typing import Tuple
import torch
from torch import nn
from modules.blocks.residual import ResidualStack


class ResNetBackbone(nn.Module):
    """ResNet backbone implementation using ResidualStack."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        filters: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        activation: str = "relu",
        noisy_sigma: float = 0.0,
        norm_type: str = "none",
    ):
        super().__init__()
        self.input_shape = input_shape

        self.stack = ResidualStack(
            input_shape=input_shape,
            filters=filters,
            kernel_sizes=kernel_sizes,
            strides=strides,
            activation=activation,
            noisy_sigma=noisy_sigma,
            norm_type=norm_type,
        )

        # Calculate output shape (C, H, W)
        # ResidualStack uses padding='same', so spatial size only changes with stride
        curr_h, curr_w = input_shape[1], input_shape[2]
        for stride in strides:
            curr_h = (curr_h + stride - 1) // stride
            curr_w = (curr_w + stride - 1) // stride

        self.output_shape = (self.stack.output_channels, curr_h, curr_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)

    def reset_noise(self) -> None:
        self.stack.reset_noise()
