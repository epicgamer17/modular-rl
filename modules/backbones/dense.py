from typing import Tuple, List, Literal, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from modules.utils import build_normalization_layer
from configs.modules.backbones.dense import DenseConfig

def build_dense(
    in_features: int, out_features: int, sigma: float = 0, bias: bool = True
) -> nn.Module:
    """Helper to create either a standard Linear layer or a NoisyDense layer."""
    if sigma == 0:
        return nn.Linear(in_features, out_features, bias=bias)
    else:
        return NoisyDense(in_features, out_features, initial_sigma=sigma, bias=bias)


class NoisyDense(nn.Module):
    """See https://arxiv.org/pdf/1706.10295."""

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
        if self.use_factorized:
            eps_i = torch.randn(1, self.in_features).to(self.mu_w.device)
            eps_j = torch.randn(self.out_features, 1).to(self.mu_w.device)
            self.eps_w = self.f(eps_j) @ self.f(eps_i)
            self.eps_b = self.f(eps_j).reshape(self.out_features) if self.use_bias else None
        else:
            self.eps_w = self.f(torch.randn(self.mu_w.shape)).to(self.mu_w.device)
            if self.use_bias:
                self.eps_b = self.f(torch.randn(size=self.mu_b.shape)).to(self.mu_w.device)

    def remove_noise(self) -> None:
        self.eps_w = torch.zeros_like(self.mu_w).to(self.mu_w.device)
        if self.use_bias:
            self.eps_b = torch.zeros_like(self.mu_b).to(self.mu_w.device)

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


class DenseStack(nn.Module):
    def __init__(
        self,
        initial_width: int,
        widths: List[int],
        activation: nn.Module = nn.ReLU(),
        noisy_sigma: float = 0,
        norm_type: Literal["batch", "layer", "none"] = "none",
    ):
        super().__init__()
        self.activation = activation
        self.noisy = noisy_sigma != 0
        self._layers = nn.ModuleList()

        current_input_width = initial_width
        for width in widths:
            use_bias = norm_type == "none"
            if noisy_sigma == 0:
                dense_layer = nn.Linear(current_input_width, width, bias=use_bias)
            else:
                dense_layer = NoisyDense(current_input_width, width, initial_sigma=noisy_sigma, bias=use_bias)

            norm_layer = build_normalization_layer(norm_type, width, dim=1)
            layer = nn.Sequential(dense_layer, norm_layer)
            self._layers.append(layer)
            current_input_width = width

        self.output_width = current_input_width

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


class DenseBackbone(nn.Module):
    """Dense (MLP) backbone implementation."""

    def __init__(self, config: DenseConfig, input_shape: Tuple[int, ...]):
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

        self.stack = DenseStack(
            initial_width=initial_width,
            widths=config.widths,
            activation=config.activation,
            noisy_sigma=config.noisy_sigma,
            norm_type=config.norm_type,
        )

        self.output_shape = (self.stack.output_width,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for a feature extraction backbone."""
        assert x.dim() == 2, f"DenseBackbone input must be (Batch, Features), got shape {x.shape}"
        return self.stack(x)

    def reset_noise(self) -> None:
        self.stack.reset_noise()
