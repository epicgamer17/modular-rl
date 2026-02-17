from collections.abc import Callable
from typing import Tuple

import torch
from modules.conv import Conv2dStack
from modules.dense import DenseStack
from modules.residual import ResidualStack
from configs.base import Config
from torch import nn

from utils.utils import to_lists
from modules.utils import prepare_activations


class NetworkBlock(nn.Module):
    """
    A reusable module to define a sequence of Residual, Conv, and Dense layers.
    It handles initialization and output shape calculation.
    """

    def __init__(
        self, config: Config, input_shape: Tuple[int], layer_prefix: str = "default"
    ):
        super().__init__()
        self.config = config
        self.input_shape = input_shape
        current_shape = input_shape
        B = current_shape[0]  # Batch size is kept symbolic

        # Dynamically fetch layer configurations
        if layer_prefix == "default":
            residual_layers = getattr(config, f"residual_layers", [])
            conv_layers = getattr(config, f"conv_layers", [])
            dense_layer_widths = getattr(config, f"dense_layer_widths", [])
        else:
            residual_layers = getattr(config, f"{layer_prefix}_residual_layers", [])
            conv_layers = getattr(config, f"{layer_prefix}_conv_layers", [])
            dense_layer_widths = getattr(
                config, f"{layer_prefix}_dense_layer_widths", []
            )

        self.has_residual_layers = len(residual_layers) > 0
        self.has_conv_layers = len(conv_layers) > 0
        self.has_dense_layers = len(dense_layer_widths) > 0

        # 1. Residual Layers
        if self.has_residual_layers:
            assert (
                len(current_shape) == 4
            ), f"{layer_prefix} residual layers expects (B, C, H, W), got {current_shape}"
            filters, kernel_sizes, strides = to_lists(residual_layers)
            self.residual_layers = ResidualStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=config.activation,
                noisy_sigma=config.noisy_sigma,
                norm_type=config.norm_type,
            )
            current_shape = (
                B,
                self.residual_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )
        else:
            self.residual_layers = nn.Identity()

        # 2. Conv Layers
        if self.has_conv_layers:
            assert (
                len(current_shape) == 4
            ), f"{layer_prefix} conv layers expects (B, C, H, W), got {current_shape}"
            filters, kernel_sizes, strides = to_lists(conv_layers)
            self.conv_layers = Conv2dStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=config.activation,
                noisy_sigma=config.noisy_sigma,
                norm_type=config.norm_type,
            )
            current_shape = (
                B,
                self.conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )
        else:
            self.conv_layers = nn.Identity()

        # 3. Dense Layers
        if self.has_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert len(current_shape) == 2
                initial_width = current_shape[1]

            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=dense_layer_widths,
                activation=config.activation,
                noisy_sigma=config.noisy_sigma,
                norm_type=config.norm_type,
            )
            current_shape = (B, self.dense_layers.output_width)
        else:
            self.dense_layers = nn.Identity()

        self.output_shape = current_shape

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        if self.has_residual_layers:
            self.residual_layers.initialize(initializer)
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual
        x = self.residual_layers(x)
        # Conv
        x = self.conv_layers(x)

        # Dense
        if self.has_dense_layers:
            x = x.flatten(1, -1)
            x = self.dense_layers(x)

        return x

    def reset_noise(self) -> None:
        if self.has_residual_layers:
            self.residual_layers.reset_noise()
        if self.has_conv_layers:
            self.conv_layers.reset_noise()
        if self.has_dense_layers:
            self.dense_layers.reset_noise()
