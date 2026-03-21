from typing import Tuple
import torch
from torch import nn
from modules.blocks.dense import build_dense
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


    def reset_noise(self) -> None:
        if hasattr(self.linear, "reset_noise"):
            self.linear.reset_noise()


class DenseResNetBackbone(nn.Module):
    """DenseResNet backbone implementation (MLP with residual connections)."""

    def __init__(self, config: DenseResNetConfig, input_shape: Tuple[int, ...]):
        super().__init__()
        self.config = config
        self.input_shape = input_shape

        # Determine initial width
        if len(input_shape) == 3:
            # Flattened image input (C, H, W)
            initial_width = input_shape[0] * input_shape[1] * input_shape[2]
        else:
            # Vector input (D,)
            initial_width = input_shape[0]

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

        self.output_shape = (current_width,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for a feature extraction backbone."""
        # --- STRICT FEATURE EXTRACTOR CONTRACT ---
        # Feature extractors always treat (B*, ...) flat batches.
        # For DenseResNetBackbone, this means (B*, D)
        assert x.dim() == 2, f"DenseResNetBackbone input must be (Batch, Features), got shape {x.shape}"

        for layer in self.layers:
            x = layer(x)
        return x


    def reset_noise(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "reset_noise"):
                layer.reset_noise()
            elif isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if hasattr(sublayer, "reset_noise"):
                        sublayer.reset_noise()
