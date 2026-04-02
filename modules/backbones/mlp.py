# modules/backbones/mlp.py
from typing import Literal, Tuple, List

import torch
from torch import nn

from modules.layers.noisy_linear import build_linear_layer
from modules.utils import build_normalization_layer


class MLPBackbone(nn.Module):
    """MLP backbone that builds linear layers directly."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        widths: List[int],
        activation: nn.Module = nn.ReLU(),
        noisy_sigma: float = 0.0,
        norm_type: Literal["batch", "layer", "none"] = "none",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.activation = activation

        # Determine initial width
        if len(input_shape) == 3:
            initial_width = input_shape[0] * input_shape[1] * input_shape[2]
        else:
            initial_width = input_shape[0]

        self._layers = nn.ModuleList()
        current_width = initial_width

        for width in widths:
            use_bias = norm_type == "none"
            linear = build_linear_layer(
                in_features=current_width,
                out_features=width,
                sigma=noisy_sigma,
                bias=use_bias,
            )
            norm = build_normalization_layer(norm_type, width, dim=1)
            self._layers.append(nn.Sequential(linear, norm))
            current_width = width

        self.output_shape = (current_width,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.flatten(1, -1)
        for layer in self._layers:
            x = self.activation(layer(x))
        return x
