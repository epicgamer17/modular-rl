from typing import Tuple, Optional
from torch import nn
from .base import BackboneConfig


class DeconvConfig(BackboneConfig):
    """Configuration for Deconvolutional (Up-sampling) backbone."""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.filters: list[int] = self.parse_field("filters", [64, 32, 3])
        self.kernel_sizes: list[int | Tuple[int, int]] = self.parse_field(
            "kernel_sizes", [3, 3, 3]
        )
        self.strides: list[int | Tuple[int, int]] = self.parse_field(
            "strides", [2, 2, 2]
        )
        self.output_padding: list[int | Tuple[int, int]] = self.parse_field(
            "output_padding", [0, 0, 0]
        )
        self.activation: nn.Module = self.parse_field("activation", nn.ReLU())
        self.norm_type: str = self.parse_field("norm_type", "none")
