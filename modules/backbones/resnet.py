# modules/backbones/resnet.py
from typing import Literal, Tuple, List

import torch
from torch import nn

from modules.layers.residual_block import ResidualBlock
from modules.utils import unpack


class ResNetBackbone(nn.Module):
    """ResNet backbone that builds ResidualBlock layers directly."""

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

        assert len(filters) == len(kernel_sizes) == len(strides), (
            f"Length mismatch: filters({len(filters)}), "
            f"kernel_sizes({len(kernel_sizes)}), strides({len(strides)})"
        )

        blocks = []
        current_channels = input_shape[0]
        curr_h, curr_w = input_shape[1], input_shape[2]

        for i in range(len(filters)):
            stride = unpack(strides[i])[0]
            block = ResidualBlock(
                in_channels=current_channels,
                out_channels=filters[i],
                kernel_size=kernel_sizes[i],
                stride=stride,
                norm_type=norm_type,
                activation=activation,
                input_size=(curr_h, curr_w),
            )
            blocks.append(block)
            current_channels = filters[i]
            curr_h = (curr_h + stride - 1) // stride
            curr_w = (curr_w + stride - 1) // stride

        self._layers = nn.Sequential(*blocks)
        self.output_shape = (current_channels, curr_h, curr_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)
