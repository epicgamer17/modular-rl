# modules/layers/residual_block.py
from typing import Literal, Tuple

from torch import nn, Tensor
from modules.utils import build_normalization_layer, calculate_same_padding


class ResidualBlock(nn.Module):
    """
    A single Residual Block (two Conv2d layers with skip connection).
    Normalization type is configurable.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        norm_type: Literal["batch", "layer", "none"] = "batch",
        activation: nn.Module = nn.ReLU(),
        input_size: Tuple[int, int] = None,
    ):
        super().__init__()
        self.activation = activation

        if input_size is not None:
            self.manual_padding, self.torch_padding = calculate_same_padding(
                input_size, kernel_size, stride
            )
        else:
            self.manual_padding, self.torch_padding = None, "same"

        # 1st Conv + Norm
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=self.torch_padding if self.torch_padding is not None else 0,
            bias=(norm_type == "none"),
        )
        self.pad1 = (
            nn.ZeroPad2d(self.manual_padding)
            if self.manual_padding is not None
            else nn.Identity()
        )
        self.norm1 = build_normalization_layer(norm_type, out_channels, dim=2)

        # 2nd Conv + Norm
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding="same",
            bias=(norm_type == "none"),
        )
        self.norm2 = build_normalization_layer(norm_type, out_channels, dim=2)

        # Downsample for skip connection if channels change
        self.downsample = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=(norm_type == "none"),
                ),
                build_normalization_layer(norm_type, out_channels, dim=2),
            )

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.downsample(inputs)

        x = self.pad1(inputs)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = self.activation(x + residual)
        return x
