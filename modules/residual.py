from typing import Callable, Literal, Tuple, Union

from torch import nn, Tensor
from modules.utils import calculate_padding

# modules/residual_block.py (New File)
from torch import nn, Tensor
from modules.utils import build_normalization_layer, calculate_same_padding, unpack


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


# modules/residual_stack.py
from torch import nn, Tensor
from modules.base_stack import BaseStack


class ResidualStack(BaseStack):
    def __init__(
        self,
        input_shape: tuple[int],
        filters: list[int],
        kernel_sizes: list[int],  # <--- Changed from kernel_size: int
        strides: list[Union[int, Tuple[int, int]]],
        activation: nn.Module = nn.ReLU(),
        noisy_sigma: float = 0,
        norm_type: Literal["batch", "layer", "none"] = "batch",
    ):
        super().__init__(activation=activation, noisy_sigma=noisy_sigma)

        # Sanity check to ensure configuration lists align
        assert len(filters) == len(kernel_sizes) == len(strides), (
            f"Length mismatch: filters({len(filters)}), "
            f"kernel_sizes({len(kernel_sizes)}), strides({len(strides)})"
        )

        current_input_channels = input_shape[1]
        curr_h, curr_w = input_shape[2], input_shape[3]

        for i in range(len(filters)):
            out_channels = filters[i]
            k_size = kernel_sizes[
                i
            ]  # <--- Extract the specific kernel size for this layer
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

        self._output_len = current_input_channels

    @property
    def output_channels(self) -> int:
        """Returns the number of output channels (C) from the final block."""
        return self._output_len

    def forward(self, inputs):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x


# class ResidualStack(nn.Module):
#     def __init__(
#         self,
#         input_shape: tuple[int],
#         filters: list[int],
#         kernel_sizes: list[int | Tuple[int, int]],
#         strides: list[int | Tuple[int, int]],
#         activation: nn.Module = nn.ReLU(),
#         noisy_sigma: float = 0,
#     ):
#         """A sequence of residual layers with the activation function applied after each layer.
#         Always applies the minimum zero-padding that ensures the output shape is equal to the input shape.
#         Input shape in "BCHW" form, i.e. (batch_size, input_channels, height, width)
#         """
#         super(ResidualStack, self).__init__()
#         self.residual_layers = nn.ModuleList()

#         self.activation = activation

#         # [B, C_in, H, W]
#         assert (
#             len(input_shape) == 4
#             and len(filters) == len(kernel_sizes) == len(strides)
#             and len(filters) > 0
#         )

#         self.noisy = noisy_sigma != 0
#         if self.noisy:
#             print("warning: Noisy convolutions not implemented yet")
#             # raise NotImplementedError("")

#         current_input_channels = input_shape[1]

#         for i in range(len(filters)):
#             print(current_input_channels)
#             layer = Residual(
#                 in_channels=current_input_channels,
#                 out_channels=filters[i],
#                 kernel_size=kernel_sizes[i],
#                 stride=strides[i],
#             )
#             self.residual_layers.append(layer)
#             current_input_channels = filters[i]

#         self._output_len = current_input_channels

#     def initialize(self, initializer: Callable[[Tensor], None]) -> None:
#         def initialize_if_conv(m: nn.Module):
#             if isinstance(m, nn.Conv2d):
#                 initializer(m.weight)

#         self.apply(initialize_if_conv)

#     def forward(self, inputs):
#         x = inputs
#         for layer in self.residual_layers:
#             x = self.activation(layer(x))
#         return x

#     def reset_noise(self):
#         assert self.noisy

#         # noisy not implemented

#         # for layer in self.conv_layers:
#         #     # layer.reset_noise()
#         # return

#     def remove_noise(self):
#         assert self.noisy

#         # noisy not implemented

#         # for layer in self.conv_layers:
#         #     # layer.reset_noise()
#         # return

#     @property
#     def output_channels(self):
#         return self._output_len


# class Residual(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride,
#     ):
#         super(Residual, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding="same",
#             bias=False,
#         )

#         # REGULARIZATION?
#         self.bn1 = nn.BatchNorm2d(
#             num_features=out_channels,
#         )

#         self.conv2 = nn.Conv2d(
#             in_channels=out_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding="same",
#             bias=False,
#         )

#         # REGULARIZATION?
#         self.bn2 = nn.BatchNorm2d(
#             num_features=out_channels,
#         )

#         self.relu = nn.ReLU()
#         self.downsample = None
#         if in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     kernel_size=kernel_size,
#                     padding="same",
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(out_channels),
#             )

#     def initialize(self, initializer: Callable[[Tensor], None]) -> None:
#         def initialize_if_conv(m: nn.Module):
#             if isinstance(m, nn.Conv2d):
#                 initializer(m.weight)

#         self.apply(initialize_if_conv)

#     def forward(self, inputs):
#         residual = self.downsample(inputs) if self.downsample else inputs

#         x = self.conv1(inputs)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x + residual)
#         return x
