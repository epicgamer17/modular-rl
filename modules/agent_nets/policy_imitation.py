from typing import Callable
from torch import Tensor
from utils.utils import to_lists
from modules.conv import Conv2dStack

from modules.dense import DenseStack, build_dense
from modules.residual import ResidualStack
import torch.nn as nn


class SupervisedNetwork(nn.Module):
    def __init__(self, config, output_size, input_shape, *args, **kwargs):
        super().__init__()
        self.config = config
        self.has_residual_layers = len(config.residual_layers) > 0
        self.has_conv_layers = len(config.conv_layers) > 0
        self.has_dense_layers = len(config.dense_layers_widths) > 0
        assert (
            self.has_conv_layers or self.has_dense_layers or self.has_residual_layers
        ), "At least one of the layers should be present."

        self.output_size = output_size

        current_shape = input_shape
        B = current_shape[0]

        if self.has_residual_layers:
            assert (
                len(current_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
            filters, kernel_sizes, strides = to_lists(config.residual_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.residual_layers = ResidualStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.residual_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_conv_layers:
            assert (
                len(current_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
            filters, kernel_sizes, strides = to_lists(config.conv_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.conv_layers = Conv2dStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert (
                    len(current_shape) == 2
                ), "Input shape should be (B, width), got {}".format(current_shape)
                initial_width = current_shape[1]

            # (B, width_in) -> (B, width_out)
            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.dense_layers_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )
            current_shape = (
                B,
                self.dense_layers.output_width,
            )

        if len(current_shape) == 4:
            initial_width = current_shape[1] * current_shape[2] * current_shape[3]
        else:
            assert len(current_shape) == 2
            initial_width = current_shape[1]

        self.output_layer = build_dense(
            initial_width,
            output_size,
            sigma=self.config.noisy_sigma,
        )
        self.return_logits = getattr(config, "return_logits", False)

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if self.has_residual_layers:
            self.residual_layers.initialize(initializer)
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)
        self.output_layer.initialize(
            initializer
        )  # OUTPUT LAYER TO IMPLIMENT INTIALIZING WITH CONSTANT OF 0.01

    def forward(self, inputs: Tensor):
        if self.has_conv_layers:
            assert inputs.dim() == 4

        x = inputs
        if self.has_residual_layers:
            x: Tensor = self.residual_layers(x)
        # print(x.shape)
        if self.has_conv_layers:
            x: Tensor = self.conv_layers(x)

        x = x.flatten(1, -1)

        if self.has_dense_layers:
            x: Tensor = self.dense_layers(x)
        x: Tensor = self.output_layer(x).view(-1, self.output_size)

        if self.return_logits:
            return x
        return x.softmax(dim=-1)

    def reset_noise(self):
        if self.has_residual_layers and hasattr(self.residual_layers, "reset_noise"):
            self.residual_layers.reset_noise()
        if self.has_conv_layers and hasattr(self.conv_layers, "reset_noise"):
            self.conv_layers.reset_noise()
        if self.has_dense_layers and hasattr(self.dense_layers, "reset_noise"):
            self.dense_layers.reset_noise()
        if hasattr(self.output_layer, "reset_noise"):
            self.output_layer.reset_noise()
