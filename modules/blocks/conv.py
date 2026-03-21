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
from modules.blocks.base_stack import BaseStack
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
            use_bias = norm_type == "none"
            conv = nn.Conv2d(
                in_channels=current_input_channels,
                out_channels=filters[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=(
                    torch_padding if not torch_padding is None else 0
                ),  # Use 0 if manual
                bias=use_bias,
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

            use_bias = norm_type == "none"
            conv = nn.ConvTranspose2d(
                in_channels=current_input_channels,
                out_channels=filters[i],
                kernel_size=k,
                stride=s,
                padding=0,  # Defaulting to valid as per common Dreamer impl
                output_padding=op,
                bias=use_bias,
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


