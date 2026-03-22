from typing import Tuple, List, Literal, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from modules.utils import build_normalization_layer
from configs.modules.backbones.mlp import MLPConfig


def build_dense(
    in_features: int, out_features: int, sigma: float = 0, bias: bool = True
) -> nn.Module:
    """Helper to create either a standard Linear layer or a NoisyLinear layer."""
    if sigma == 0:
        return nn.Linear(in_features, out_features, bias=bias)
    else:
        return NoisyLinear(in_features, out_features, initial_sigma=sigma, bias=bias)


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration. See https://arxiv.org/pdf/1706.10295."""

    @staticmethod
    def f(x: Tensor):
        return x.sgn() * (x.abs().sqrt())

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        initial_sigma: float = 0.5,
        use_factorized: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial_sigma = initial_sigma
        self.use_factorized = use_factorized
        self.use_bias = bias

        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("eps_w", torch.empty(out_features, in_features))

        if self.use_bias:
            self.mu_b = nn.Parameter(torch.empty(out_features))
            self.sigma_b = nn.Parameter(torch.empty(out_features))
            self.register_buffer("eps_b", torch.empty(out_features))
        else:
            self.register_parameter("mu_b", None)
            self.register_parameter("sigma_b", None)
            self.register_buffer("eps_b", None)

        self.reset_parameters()
        self.reset_noise()

    def reset_noise(self) -> None:
        device = self.mu_w.device
        if self.use_factorized:
            eps_i = torch.randn(1, self.in_features, device=device)
            eps_j = torch.randn(self.out_features, 1, device=device)
            self.eps_w = self.f(eps_j) @ self.f(eps_i)
            self.eps_b = (
                self.f(eps_j).reshape(self.out_features) if self.use_bias else None
            )
        else:
            self.eps_w = self.f(torch.randn(self.mu_w.shape, device=device))
            if self.use_bias:
                self.eps_b = self.f(torch.randn(size=self.mu_b.shape, device=device))

    def remove_noise(self) -> None:
        self.eps_w = torch.zeros_like(self.mu_w)
        if self.use_bias:
            self.eps_b = torch.zeros_like(self.mu_b)

    def reset_parameters(self) -> None:
        p = self.in_features
        if self.use_factorized:
            mu_init = 1.0 / (p**0.5)
            sigma_init = self.initial_sigma / (p**0.5)
        else:
            mu_init = (3.0 / p) ** 0.5
            sigma_init = 0.017

        nn.init.constant_(self.sigma_w, sigma_init)
        nn.init.uniform_(self.mu_w, -mu_init, mu_init)
        if self.use_bias:
            nn.init.constant_(self.sigma_b, sigma_init)
            nn.init.uniform_(self.mu_b, -mu_init, mu_init)

    @property
    def weight(self):
        return self.mu_w + self.sigma_w * self.eps_w

    @property
    def bias(self):
        if self.use_bias:
            return self.mu_b + self.sigma_b * self.eps_b
        return None

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


class MLPBackbone(nn.Module):
    """Dense (MLP) backbone implementation."""

    def __init__(self, config: MLPConfig, input_shape: Tuple[int, ...]):
        super().__init__()
        self.config = config
        self.input_shape = input_shape
        self.noisy = config.noisy_sigma != 0

        # Calculate dimensions sequence
        input_dim = torch.Size(input_shape).numel()
        all_dims = [input_dim] + list(config.widths)

        layers = []
        for in_f, out_f in zip(all_dims[:-1], all_dims[1:]):
            # Use bias only if not using normalization
            use_bias = config.norm_type == "none"

            # Build layer (standard or noisy)
            layers.append(
                build_dense(in_f, out_f, sigma=config.noisy_sigma, bias=use_bias)
            )

            # Optional normalization
            if config.norm_type != "none":
                layers.append(build_normalization_layer(config.norm_type, out_f, dim=1))

            # Activation
            layers.append(config.activation)

        self.model = nn.Sequential(*layers)
        self.output_shape = (all_dims[-1],)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass with automatic flattening for multi-dim inputs."""
        if x.dim() > 2:
            # Flatten everything except batch into a feature vector
            x = x.flatten(1, -1)

        return self.model(x)

    def reset_noise(self) -> None:
        if not self.noisy:
            return
        for m in self.model.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
