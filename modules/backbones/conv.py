# modules/backbones/conv.py
from typing import Literal, Tuple, List

import torch
from torch import nn

from modules.utils import build_normalization_layer, calculate_same_padding, unpack


class ConvBackbone(nn.Module):
    """Standard convolutional backbone that builds Conv2d layers directly."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        filters: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        activation: nn.Module = nn.ReLU(),
        noisy_sigma: float = 0.0,
        norm_type: Literal["batch", "layer", "none"] = "none",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.activation = activation

        self._layers = nn.ModuleList()
        current_channels = input_shape[0]
        curr_h, curr_w = input_shape[1], input_shape[2]

        for i in range(len(filters)):
            manual_padding, torch_padding = calculate_same_padding(
                (curr_h, curr_w), kernel_sizes[i], strides[i]
            )

            use_bias = norm_type == "none"
            conv = nn.Conv2d(
                in_channels=current_channels,
                out_channels=filters[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=torch_padding if torch_padding is not None else 0,
                bias=use_bias,
            )
            norm = build_normalization_layer(norm_type, filters[i], dim=2)

            if manual_padding is None:
                layer = nn.Sequential(conv, norm)
            else:
                layer = nn.Sequential(nn.ZeroPad2d(manual_padding), conv, norm)

            self._layers.append(layer)
            current_channels = filters[i]

            curr_h = (curr_h + strides[i] - 1) // strides[i]
            curr_w = (curr_w + strides[i] - 1) // strides[i]

        self.output_shape = (current_channels, curr_h, curr_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = self.activation(layer(x))
        return x


class DeconvBackbone(nn.Module):
    """Backbone for upscaling/decoding tasks (e.g. Dreamer decoder)."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        filters: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        activation: nn.Module = nn.ReLU(),
        norm_type: Literal["batch", "layer", "none"] = "none",
        output_padding: List[int] = None,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.activation = activation

        if output_padding is None:
            output_padding = [0] * len(filters)

        self._layers = nn.ModuleList()
        current_channels = input_shape[0]

        for i in range(len(filters)):
            k = unpack(kernel_sizes[i])
            s = unpack(strides[i])
            op = unpack(output_padding[i])

            use_bias = norm_type == "none"
            conv = nn.ConvTranspose2d(
                in_channels=current_channels,
                out_channels=filters[i],
                kernel_size=k,
                stride=s,
                padding=0,
                output_padding=op,
                bias=use_bias,
            )
            norm = build_normalization_layer(norm_type, filters[i], dim=2)
            self._layers.append(nn.Sequential(conv, norm))
            current_channels = filters[i]

        self.output_shape = self._compute_output_shape()

    def _compute_output_shape(self) -> Tuple[int, ...]:
        """Dummy forward pass to determine output shape."""
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            out = self.forward(dummy)
            return tuple(out.shape[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = self.activation(layer(x))
        return x
