from typing import Tuple
import torch
from torch import nn
from modules.blocks.conv import Conv2dStack


class ConvBackbone(nn.Module):
    """Standard Convolutional backbone implementation using Conv2dStack."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        filters: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        activation: nn.Module = nn.ReLU(),
        noisy_sigma: float = 0.0,
        norm_type: str = "none",
    ):
        super().__init__()
        self.input_shape = input_shape

        self.stack = Conv2dStack(
            input_shape=input_shape,
            filters=filters,
            kernel_sizes=kernel_sizes,
            strides=strides,
            activation=activation,
            noisy_sigma=noisy_sigma,
            norm_type=norm_type,
        )

        # Calculate output shape (C, H, W)
        # Conv2dStack uses padding='same' (via calculate_same_padding)
        curr_h, curr_w = input_shape[1], input_shape[2]
        for stride in strides:
            curr_h = (curr_h + stride - 1) // stride
            curr_w = (curr_w + stride - 1) // stride

        self.output_shape = (self.stack.output_channels, curr_h, curr_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)

    def reset_noise(self) -> None:
        self.stack.reset_noise()


class DeconvBackbone(nn.Module):
    """
    Backbone for upscaling/decoding tasks (e.g. Dreamer decoder).
    Wraps ConvTranspose2dStack.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        filters: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        activation: nn.Module = nn.ReLU(),
        norm_type: str = "none",
        output_padding: list[int] = None,
    ):
        super().__init__()
        self.input_shape = input_shape

        # Late import to avoid circular dependency
        from modules.blocks.conv import ConvTranspose2dStack

        self.stack = ConvTranspose2dStack(
            input_shape=input_shape,
            filters=filters,
            kernel_sizes=kernel_sizes,
            strides=strides,
            activation=activation,
            norm_type=norm_type,
            output_padding=output_padding,
        )

        self.output_shape = self._get_output_shape()

    def _get_output_shape(self) -> Tuple[int, ...]:
        # Dummy forward pass to determine output shape if not explicitly provided
        with torch.no_grad():
            dummy = torch.zeros(
                1, *self.input_shape, device=self.stack._layers[0][0].weight.device
            )
            out = self.stack(dummy)
            return tuple(out.shape[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)
