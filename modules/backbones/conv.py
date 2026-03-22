from typing import Tuple, Literal, Union, List, Optional
import torch
from torch import nn, Tensor
from modules.utils import build_normalization_layer, calculate_same_padding, unpack
from configs.modules.backbones.conv import ConvConfig
from configs.modules.backbones.deconv import DeconvConfig

class Conv2dStack(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        filters: list[int],
        kernel_sizes: list[int | Tuple[int, int]],
        strides: list[int | Tuple[int, int]],
        activation: nn.Module = nn.ReLU(),
        noisy_sigma: float = 0,
        norm_type: Literal["batch", "layer", "none"] = "none",
    ):
        super().__init__()
        self.activation = activation
        self.noisy = noisy_sigma != 0
        self._layers = nn.ModuleList()

        current_input_channels = input_shape[0]
        for i in range(len(filters)):
            h, w = input_shape[1], input_shape[2]
            manual_padding, torch_padding = calculate_same_padding(
                (h, w), kernel_sizes[i], strides[i]
            )

            use_bias = norm_type == "none"
            conv = nn.Conv2d(
                in_channels=current_input_channels,
                out_channels=filters[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=torch_padding if torch_padding is not None else 0,
                bias=use_bias,
            )

            norm_layer = build_normalization_layer(norm_type, filters[i], dim=2)

            if manual_padding is None:
                layer = nn.Sequential(conv, norm_layer)
            else:
                layer = nn.Sequential(
                    nn.ZeroPad2d(manual_padding),
                    conv,
                    norm_layer,
                )

            self._layers.append(layer)
            current_input_channels = filters[i]

            # Update spatial dimensions for next iteration
            stride_h, stride_w = unpack(strides[i])
            input_shape = (current_input_channels, (h + stride_h - 1) // stride_h, (w + stride_w - 1) // stride_w)

        self.output_channels = current_input_channels

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for layer in self._layers:
            x = self.activation(layer(x))
        return x

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        for layer in self._layers:
            if hasattr(layer, "reset_noise"):
                layer.reset_noise()


class ConvTranspose2dStack(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, ...],
        filters: list[int],
        kernel_sizes: list[int | Tuple[int, int]],
        strides: list[int | Tuple[int, int]],
        activation: nn.Module = nn.ReLU(),
        noisy_sigma: float = 0,
        norm_type: Literal["batch", "layer", "none"] = "none",
        output_padding: list[int | Tuple[int, int]] = None,
    ):
        super().__init__()
        self.activation = activation
        self.noisy = noisy_sigma != 0
        self._layers = nn.ModuleList()

        if output_padding is None:
            output_padding = [0] * len(filters)

        current_input_channels = input_shape[0]
        for i in range(len(filters)):
            k = unpack(kernel_sizes[i])
            s = unpack(strides[i])
            op = unpack(output_padding[i])

            use_bias = norm_type == "none"
            conv = nn.ConvTranspose2d(
                in_channels=current_input_channels,
                out_channels=filters[i],
                kernel_size=k,
                stride=s,
                padding=0,
                output_padding=op,
                bias=use_bias,
            )

            norm_layer = build_normalization_layer(norm_type, filters[i], dim=2)
            layer = nn.Sequential(conv, norm_layer)
            self._layers.append(layer)
            current_input_channels = filters[i]

        self.output_channels = current_input_channels

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for layer in self._layers:
            x = self.activation(layer(x))
        return x

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        for layer in self._layers:
            if hasattr(layer, "reset_noise"):
                layer.reset_noise()


class ConvBackbone(nn.Module):
    """Standard Convolutional backbone implementation."""

    def __init__(self, config: ConvConfig, input_shape: Tuple[int, ...]):
        super().__init__()
        self.config = config
        self.input_shape = input_shape

        self.stack = Conv2dStack(
            input_shape=input_shape,
            filters=config.filters,
            kernel_sizes=config.kernel_sizes,
            strides=config.strides,
            activation=config.activation,
            noisy_sigma=config.noisy_sigma,
            norm_type=config.norm_type,
        )

        curr_h, curr_w = input_shape[1], input_shape[2]
        for stride_info in config.strides:
            stride_h, stride_w = unpack(stride_info)
            curr_h = (curr_h + stride_h - 1) // stride_h
            curr_w = (curr_w + stride_w - 1) // stride_w

        self.output_shape = (self.stack.output_channels, curr_h, curr_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for a feature extraction backbone."""
        assert x.dim() == len(self.input_shape) + 1, f"ConvBackbone input must be (Batch, *input_shape), got shape {x.shape}"
        return self.stack(x)

    def reset_noise(self) -> None:
        self.stack.reset_noise()


class DeconvBackbone(nn.Module):
    """Backbone for upscaling/decoding tasks."""

    def __init__(self, config: DeconvConfig, input_shape: Tuple[int, ...]):
        super().__init__()
        self.config = config
        self.input_shape = input_shape

        self.stack = ConvTranspose2dStack(
            input_shape=input_shape,
            filters=config.filters,
            kernel_sizes=config.kernel_sizes,
            strides=config.strides,
            activation=config.activation,
            norm_type=config.norm_type,
            output_padding=config.output_padding,
        )

        self.output_shape = self._get_output_shape()

    def _get_output_shape(self) -> Tuple[int, ...]:
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape, device=self.parameters().__next__().device if any(self.parameters()) else torch.device("cpu"))
            out = self.stack(dummy)
            return tuple(out.shape[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for a feature extraction backbone."""
        assert x.dim() == len(self.input_shape) + 1, f"DeconvBackbone input must be (Batch, *input_shape), got shape {x.shape}"
        return self.stack(x)
