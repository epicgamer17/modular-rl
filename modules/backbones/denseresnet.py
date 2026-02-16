from typing import Tuple
import torch
from torch import nn
from modules.dense import build_dense
from configs.modules.backbones.denseresnet import DenseResNetConfig
from modules.utils import build_normalization_layer


class DenseResidualBlock(nn.Module):
    """A single Dense Residual Block (Linear + Norm + Skip)."""

    def __init__(
        self, size: int, activation: nn.Module, norm_type: str, noisy_sigma: float
    ):
        super().__init__()
        self.activation = activation
        self.linear = build_dense(size, size, sigma=noisy_sigma)
        self.norm = build_normalization_layer(norm_type, size, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear(x)
        x = self.norm(x)
        return self.activation(x + residual)

    def initialize(self, initializer: torch.Tensor) -> None:
        if hasattr(self.linear, "initialize"):
            self.linear.initialize(initializer)
        else:
            initializer(self.linear.layer.weight)


class DenseResNetBackbone(nn.Module):
    """DenseResNet backbone implementation (MLP with residual connections)."""

    def __init__(self, config: DenseResNetConfig, input_shape: Tuple[int, ...]):
        super().__init__()
        self.config = config
        self.input_shape = input_shape

        # Determine initial width
        if len(input_shape) == 4:
            initial_width = input_shape[1] * input_shape[2] * input_shape[3]
        else:
            initial_width = input_shape[1]

        self.layers = nn.ModuleList()
        current_width = initial_width

        for width in config.widths:
            # If width changes, add a projection layer before blocks
            if width != current_width:
                self.layers.append(
                    nn.Sequential(
                        build_dense(current_width, width, sigma=config.noisy_sigma),
                        build_normalization_layer(config.norm_type, width, dim=1),
                        config.activation,
                    )
                )
                current_width = width

            # Add a residual block
            self.layers.append(
                DenseResidualBlock(
                    size=current_width,
                    activation=config.activation,
                    norm_type=config.norm_type,
                    noisy_sigma=config.noisy_sigma,
                )
            )

        self.output_shape = (input_shape[0], current_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.flatten(1, -1)

        for layer in self.layers:
            x = layer(x)
        return x

    def initialize(self, initializer: torch.Tensor) -> None:
        for layer in self.layers:
            if hasattr(layer, "initialize"):
                layer.initialize(initializer)
            elif isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if hasattr(sublayer, "initialize"):
                        sublayer.initialize(initializer)
                    elif isinstance(sublayer, (nn.Linear, nn.Conv2d)):
                        initializer(sublayer.weight)

    def reset_noise(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "reset_noise"):
                layer.reset_noise()
