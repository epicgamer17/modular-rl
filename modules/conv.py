from typing import Callable, Literal, Tuple

from torch import nn, Tensor
from modules.utils import build_normalization_layer, calculate_padding


def unpack(x: int | Tuple):
    if isinstance(x, Tuple):
        assert len(x) == 2
        return x
    else:
        try:
            x = int(x)
            return x, x
        except Exception as e:
            print(f"error converting {x} to int: ", e)


# modules/conv2d_stack.py
from typing import Callable, Tuple
from torch import nn, Tensor
from modules.base_stack import BaseStack
from modules.utils import (
    calculate_same_padding,
    unpack,
)  # Import utility


class Conv2dStack(BaseStack):
    def __init__(
        self,
        input_shape: tuple[int],
        filters: list[int],
        kernel_sizes: list[int | Tuple[int, int]],
        strides: list[int | Tuple[int, int]],
        activation: nn.Module = nn.ReLU(),
        noisy_sigma: float = 0,
        norm_type: Literal["batch", "layer", "none"] = "none",
    ):
        super().__init__(activation=activation, noisy_sigma=noisy_sigma)

        self.norm_type = norm_type
        self.input_shape = input_shape
        # ... (assertions)

        current_input_channels = input_shape[0]
        for i in range(len(filters)):

            # Use utility for padding
            h, w = input_shape[1], input_shape[2]
            manual_padding, torch_padding = calculate_same_padding(
                (h, w), kernel_sizes[i], strides[i]
            )

            # --- START: Building the Layer ---
            conv = nn.Conv2d(
                in_channels=current_input_channels,
                out_channels=filters[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=(
                    torch_padding if not torch_padding is None else 0
                ),  # Use 0 if manual
            )

            norm_layer = build_normalization_layer(norm_type, filters[i], dim=2)

            if manual_padding is None:
                layer = nn.Sequential(
                    conv, norm_layer
                )  # Conv -> Norm -> Activation in forward
            else:
                layer = nn.Sequential(
                    nn.ZeroPad2d(manual_padding),
                    conv,
                    norm_layer,  # Pad -> Conv -> Norm -> Activation in forward
                )
            # --- END: Building the Layer ---

            self._layers.append(layer)
            current_input_channels = filters[i]

        self._output_len = current_input_channels

    @property
    def output_channels(self) -> int:
        """Returns the number of output channels (C) from the final block."""
        return self._output_len

    def forward(self, inputs):
        x = inputs
        for layer in self._layers:
            # Note: We apply activation AFTER the Conv/Norm block
            x = self.activation(layer(x))
        return x


class ConvTranspose2dStack(BaseStack):
    def __init__(
        self,
        input_shape: tuple[int],
        filters: list[int],
        kernel_sizes: list[int | Tuple[int, int]],
        strides: list[int | Tuple[int, int]],
        activation: nn.Module = nn.ReLU(),
        noisy_sigma: float = 0,
        norm_type: Literal["batch", "layer", "none"] = "none",
        output_padding: list[
            int | Tuple[int, int]
        ] = None,  # Added for Transpose specific control
    ):
        super().__init__(activation=activation, noisy_sigma=noisy_sigma)

        self.norm_type = norm_type
        self.input_shape = input_shape

        if output_padding is None:
            output_padding = [0] * len(filters)

        current_input_channels = input_shape[0]
        for i in range(len(filters)):

            # Note: Padding calculation for Transpose is inverse of Conv.
            # Dreamer uses 'valid' (0 padding) usually or specific sizes.
            # Here we expose standard params.

            k = unpack(kernel_sizes[i])
            s = unpack(strides[i])
            op = unpack(output_padding[i])

            # Dreamer TF implementation often uses padding='valid' (0 in PyTorch) for Decoder
            # but relies on the kernel size to upscale.

            conv = nn.ConvTranspose2d(
                in_channels=current_input_channels,
                out_channels=filters[i],
                kernel_size=k,
                stride=s,
                padding=0,  # Defaulting to valid as per common Dreamer impl
                output_padding=op,
            )

            norm_layer = build_normalization_layer(norm_type, filters[i], dim=2)

            layer = nn.Sequential(conv, norm_layer)

            self._layers.append(layer)
            current_input_channels = filters[i]

        self._output_len = current_input_channels

    @property
    def output_channels(self) -> int:
        return self._output_len

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self._layers):
            x = layer(x)
            # Apply activation to all but potentially the last layer if specified externally,
            # but BaseStack pattern implies activation on all.
            # Dreamer decoder usually activates all but final distribution head.
            x = self.activation(x)
        return x


# class Conv2dStack(nn.Module):
#     @staticmethod
#     def calculate_same_padding(i, k, s) -> Tuple[None | Tuple[int], None | str | Tuple]:
#         """Calculate pytorch inputs for same padding
#         Args:
#             i (int, int) or int: (h, w) or (w, w)
#             k (int, int) or int: (k_h, k_w) or (k, k)
#             s (int, int) or int: (s_h, s_w) or (s, s)
#         Returns:
#             Tuple[manual_pad_padding, torch_conv2d_padding_input]: Either the manual padding that must be applied (first element of tuple) or the input to the torch padding argument of the Conv2d layer
#         """

#         if s == 1:
#             return None, "same"
#         h, w = unpack(i)
#         k_h, k_w = unpack(k)
#         s_h, s_w = unpack(s)
#         p_h = calculate_padding(h, k_h, s_h)
#         p_w = calculate_padding(w, k_w, s_w)
#         if p_h[0] == p_h[1] and p_w[0] == p_w[1]:
#             return None, (p_h[0], p_w[0])
#         else:
#             # not torch compatiable, manually pad with torch.nn.functional.pad
#             return (*p_w, *p_h), None

#     def __init__(
#         self,
#         input_shape: tuple[int],
#         filters: list[int],
#         kernel_sizes: list[int | Tuple[int, int]],
#         strides: list[int | Tuple[int, int]],
#         activation: nn.Module = nn.ReLU(),
#         noisy_sigma: float = 0,
#     ):
#         """A sequence of convolution layers with the activation function applied after each layer.
#         Always applies the minimum zero-padding that ensures the output shape is equal to the input shape.
#         Input shape in "BCHW" form, i.e. (batch_size, input_channels, height, width)
#         """
#         super(Conv2dStack, self).__init__()
#         self.conv_layers = nn.ModuleList()

#         self.activation = activation

#         # [B, C_in, H, W]
#         assert len(input_shape) == 4
#         assert len(filters) == len(kernel_sizes) == len(strides)
#         assert len(filters) > 0

#         self.noisy = noisy_sigma != 0
#         if self.noisy:
#             print("warning: Noisy convolutions not implemented yet")
#             # raise NotImplementedError("")

#         current_input_channels = input_shape[1]
#         for i in range(len(filters)):

#             h, w = input_shape[2], input_shape[3]
#             manual_padding, torch_padding = self.calculate_same_padding(
#                 (h, w), kernel_sizes[i], strides[i]
#             )

#             if not torch_padding is None:
#                 layer = nn.Conv2d(
#                     in_channels=current_input_channels,
#                     out_channels=filters[i],
#                     kernel_size=kernel_sizes[i],
#                     stride=strides[i],
#                     padding=torch_padding,
#                 )
#             else:
#                 layer = nn.Sequential(
#                     nn.ZeroPad2d(manual_padding),
#                     nn.Conv2d(
#                         in_channels=current_input_channels,
#                         out_channels=filters[i],
#                         kernel_size=kernel_sizes[i],
#                         stride=strides[i],
#                     ),
#                 )

#             self.conv_layers.append(layer)
#             current_input_channels = filters[i]

#         self._output_len = current_input_channels

#     def initialize(self, initializer: Callable[[Tensor], None]) -> None:
#         def initialize_if_conv(m: nn.Module):
#             if isinstance(m, nn.Conv2d):
#                 initializer(m.weight)

#         self.apply(initialize_if_conv)

#     def forward(self, inputs):
#         x = inputs
#         for layer in self.conv_layers:
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
