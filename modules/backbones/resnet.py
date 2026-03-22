from typing import Tuple, Literal, Union, List, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from modules.utils import build_normalization_layer, calculate_same_padding, unpack
from configs.modules.backbones.resnet import ResNetConfig

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
        input_size: Tuple[int, int] = None,  # (h, w)
    ):
        super().__init__()
        self.activation = activation

        # Calculate padding for same effect with stride
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
            # Use a 1x1 conv to match feature dimensions
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

        # Skip Connection
        x = self.activation(x + residual)
        return x


class ResidualStack(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        filters: list[int],
        kernel_sizes: list[int],
        strides: list[Union[int, Tuple[int, int]]],
        activation: nn.Module = nn.ReLU(),
        noisy_sigma: float = 0,
        norm_type: Literal["batch", "layer", "none"] = "batch",
    ):
        super().__init__()
        self.noisy = noisy_sigma != 0
        self._layers = nn.ModuleList()

        assert len(filters) == len(kernel_sizes) == len(strides), (
            f"Length mismatch: filters({len(filters)}), "
            f"kernel_sizes({len(kernel_sizes)}), strides({len(strides)})"
        )

        current_input_channels = input_shape[0]
        curr_h, curr_w = input_shape[1], input_shape[2]

        for i in range(len(filters)):
            out_channels = filters[i]
            k_size = kernel_sizes[i]
            stride = unpack(strides[i])[0]

            layer = ResidualBlock(
                in_channels=current_input_channels,
                out_channels=out_channels,
                kernel_size=k_size,
                stride=stride,
                norm_type=norm_type,
                activation=activation,
                input_size=(curr_h, curr_w),
            )
            self._layers.append(layer)
            current_input_channels = out_channels

            # Update spatial dimensions for next layer
            curr_h = (curr_h + stride - 1) // stride
            curr_w = (curr_w + stride - 1) // stride

        self.output_channels = current_input_channels

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        for layer in self._layers:
            if hasattr(layer, "reset_noise"):
                layer.reset_noise()


class ResNetBackbone(nn.Module):
    """ResNet backbone implementation."""

    def __init__(self, config: ResNetConfig, input_shape: Tuple[int, ...]):
        super().__init__()
        self.config = config
        self.input_shape = input_shape

        self.stack = ResidualStack(
            input_shape=input_shape,
            filters=config.filters,
            kernel_sizes=config.kernel_sizes,
            strides=config.strides,
            activation=config.activation,
            noisy_sigma=config.noisy_sigma,
            norm_type=config.norm_type,
        )

        # Calculate output shape (C, H, W)
        curr_h, curr_w = input_shape[1], input_shape[2]
        for stride_info in config.strides:
            stride = unpack(stride_info)[0]
            curr_h = (curr_h + stride - 1) // stride
            curr_w = (curr_w + stride - 1) // stride

        self.output_shape = (self.stack.output_channels, curr_h, curr_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for a feature extraction backbone."""
        assert x.dim() == len(self.input_shape) + 1, f"ResNetBackbone input must be (Batch, *input_shape), got shape {x.shape}"
        return self.stack(x)

    def reset_noise(self) -> None:
        self.stack.reset_noise()
