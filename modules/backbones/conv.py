from typing import Tuple, Literal, Union, List, Optional
import torch
from torch import nn, Tensor
from modules.utils import build_normalization_layer, calculate_same_padding, unpack
from configs.modules.backbones.conv import ConvConfig
from configs.modules.backbones.deconv import DeconvConfig


class ConvBackbone(nn.Module):
    """Standard Convolutional backbone implementation."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        filters: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        norm_type: str = "batch",
        activation: nn.Module = nn.ReLU(),
        noisy_sigma: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.noisy = noisy_sigma != 0

        layers = []
        current_input_channels = input_shape[0]
        h, w = input_shape[1], input_shape[2]
        
        for i in range(len(filters)):
            # Padding calculation
            manual_padding, torch_padding = calculate_same_padding(
                (h, w), kernel_sizes[i], strides[i]
            )

            # Bias rule: bias=False if followed by normalization
            use_bias = norm_type == "none"

            # Convolutional layer
            conv = nn.Conv2d(
                in_channels=current_input_channels,
                out_channels=filters[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                padding=torch_padding if torch_padding is not None else 0,
                bias=use_bias,
            )

            # Normalization layer
            norm_layer = build_normalization_layer(norm_type, filters[i], dim=2)

            # Construct sequential block for this stage
            if manual_padding is None:
                stage = nn.Sequential(conv, norm_layer, activation)
            else:
                stage = nn.Sequential(
                    nn.ZeroPad2d(manual_padding),
                    conv,
                    norm_layer,
                    activation,
                )

            layers.append(stage)
            current_input_channels = filters[i]

            # Update spatial dimensions for next iteration
            stride_h, stride_w = unpack(strides[i])
            h = (h + stride_h - 1) // stride_h
            w = (w + stride_w - 1) // stride_w

        self.model = nn.Sequential(*layers)
        self.output_shape = (current_input_channels, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for a feature extraction backbone."""
        assert x.dim() == len(self.input_shape) + 1, f"ConvBackbone input must be (Batch, *input_shape), got shape {x.shape}"
        return self.model(x)

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        for m in self.model.modules():
            if hasattr(m, "reset_noise"):
                m.reset_noise()


class DeconvBackbone(nn.Module):
    """Backbone for upscaling/decoding tasks."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        filters: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        output_padding: Optional[List[int]] = None,
        norm_type: str = "batch",
        activation: nn.Module = nn.ReLU(),
        **kwargs,
    ):
        super().__init__()
        self.input_shape = input_shape

        layers = []
        current_input_channels = input_shape[0]
        op_list = output_padding if output_padding is not None else [0] * len(filters)

        for i in range(len(filters)):
            k = unpack(kernel_sizes[i])
            s = unpack(strides[i])
            op = unpack(op_list[i])

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
            layers.append(nn.Sequential(conv, norm_layer, activation))
            current_input_channels = filters[i]

        self.model = nn.Sequential(*layers)
        self.output_shape = self._get_output_shape()

    def _get_output_shape(self) -> Tuple[int, ...]:
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            out = self.model(dummy)
            return tuple(out.shape[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for a feature extraction backbone."""
        assert x.dim() == len(self.input_shape) + 1, f"DeconvBackbone input must be (Batch, *input_shape), got shape {x.shape}"
        return self.model(x)
