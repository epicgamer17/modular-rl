from typing import Tuple
import torch
from torch import nn
from modules.backbones.mlp import build_dense
from configs.modules.backbones.mlpresnet import MLPResNetConfig
from modules.utils import build_normalization_layer


class MLPResidualBlock(nn.Module):
    """A single MLP Residual Block (Linear + Norm + Skip)."""

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


class MLPResNetBackbone(nn.Module):
    """MLPResNet backbone implementation (MLP with residual connections)."""

    def __init__(self, config: MLPResNetConfig, input_shape: Tuple[int, ...]):
        super().__init__()
        self.config = config
        self.input_shape = input_shape
        self.noisy = config.noisy_sigma != 0

        # Sequence of dimensions for residual stream
        input_dim = torch.Size(input_shape).numel()
        all_dims = [input_dim] + list(config.widths)

        layers = []
        for in_f, out_f in zip(all_dims[:-1], all_dims[1:]):
            # If width changes, add a projection layer before blocks
            if in_f != out_f:
                layers.append(
                    nn.Sequential(
                        build_dense(in_f, out_f, sigma=config.noisy_sigma),
                        build_normalization_layer(config.norm_type, out_f, dim=1),
                        config.activation,
                    )
                )

            # Add a residual block
            layers.append(
                MLPResidualBlock(
                    size=out_f,
                    activation=config.activation,
                    norm_type=config.norm_type,
                    noisy_sigma=config.noisy_sigma,
                )
            )

        self.model = nn.Sequential(*layers)
        self.output_shape = (all_dims[-1],)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic flattening for multi-dim inputs."""
        if x.dim() > 2:
            x = x.flatten(1, -1)

        return self.model(x)

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        for m in self.model.modules():
            if hasattr(m, "reset_noise"):
                m.reset_noise()
