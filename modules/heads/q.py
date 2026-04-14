# modules/heads/q.py
from typing import Tuple, Optional, Dict, Any, List, Literal

import torch
from torch import nn, Tensor

from .base import BaseHead
from modules.representations import BaseRepresentation
from core.contracts import ActionDistribution
from modules.layers.noisy_linear import build_linear_layer
from modules.utils import build_normalization_layer


def _build_hidden_layers(
    initial_width: int,
    widths: List[int],
    activation: nn.Module,
    noisy_sigma: float,
    norm_type: Literal["batch", "layer", "none"],
) -> Tuple[nn.ModuleList, nn.Module, int]:
    """Builds a list of hidden linear layers with norm. Returns (layers, activation, output_width)."""
    layers = nn.ModuleList()
    current_width = initial_width
    for width in widths:
        use_bias = norm_type == "none"
        linear = build_linear_layer(
            in_features=current_width,
            out_features=width,
            sigma=noisy_sigma,
            bias=use_bias,
        )
        norm = build_normalization_layer(norm_type, width, dim=1)
        layers.append(nn.Sequential(linear, norm))
        current_width = width
    return layers, activation, current_width


def _forward_hidden(layers: nn.ModuleList, activation: nn.Module, x: Tensor) -> Tensor:
    """Forward pass through hidden layers with activation."""
    for layer in layers:
        x = activation(layer(x))
    return x


class QHead(BaseHead):
    """
    Standard Q-Network Head.
    Structure: [Neck (Optional)] -> [Hidden Layers] -> [Output Layer] -> [Strategy (Reshape)]
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        hidden_widths: List[int],
        num_actions: int,
        neck: Optional[nn.Module] = None,
        noisy_sigma: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        norm_type: str = "none",
    ):
        super().__init__(input_shape, representation, neck, noisy_sigma)

        self.num_actions = num_actions
        self.input_dim = self.flat_dim

        self.hidden_layers, self._hidden_activation, hidden_out_width = (
            _build_hidden_layers(
                initial_width=self.input_dim,
                widths=hidden_widths,
                activation=activation,
                noisy_sigma=noisy_sigma,
                norm_type=norm_type,
            )
        )

        # Output Layer (Overwrites BaseHead's output_layer)
        self.output_layer = build_linear_layer(
            in_features=hidden_out_width,
            out_features=self.representation.num_features * self.num_actions,
            sigma=noisy_sigma,
        )

    def forward(
        self, x: Tensor, state: Optional[Dict[str, Any]] = None
    ) -> Tuple[Tensor, Dict[str, Any], Any]:
        x = self.process_input(x)
        x = _forward_hidden(self.hidden_layers, self._hidden_activation, x)
        logits = self.output_layer(x)
        logits = logits.view(-1, self.num_actions, self.representation.num_features)
        new_state = state if state is not None else {}
        inference = self.representation.to_inference(logits)
        return logits, new_state, inference

    @property
    def semantic_type(self) -> Any:
        return ActionDistribution[self.get_structure()]


class DuelingQHead(BaseHead):
    """
    Dueling Q-Network Head.
    Structure:
        [Neck] -> Split -> [Value Hidden] -> [Value Output]
                        -> [Advantage Hidden] -> [Advantage Output]
               -> Aggregation (Q = V + A - mean(A))
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        value_hidden_widths: List[int],
        advantage_hidden_widths: List[int],
        num_actions: int,
        neck: Optional[nn.Module] = None,
        noisy_sigma: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        norm_type: str = "none",
    ):
        super().__init__(input_shape, representation, neck, noisy_sigma)

        self.num_actions = num_actions
        self.input_dim = self.flat_dim

        # Value Stream
        self.value_hidden, self._value_activation, value_out_width = (
            _build_hidden_layers(
                initial_width=self.input_dim,
                widths=value_hidden_widths,
                activation=activation,
                noisy_sigma=noisy_sigma,
                norm_type=norm_type,
            )
        )
        self.value_output = build_linear_layer(
            in_features=value_out_width,
            out_features=self.representation.num_features,
            sigma=noisy_sigma,
        )

        # Advantage Stream
        self.advantage_hidden, self._advantage_activation, adv_out_width = (
            _build_hidden_layers(
                initial_width=self.input_dim,
                widths=advantage_hidden_widths,
                activation=activation,
                noisy_sigma=noisy_sigma,
                norm_type=norm_type,
            )
        )
        self.advantage_output = build_linear_layer(
            in_features=adv_out_width,
            out_features=self.representation.num_features * self.num_actions,
            sigma=noisy_sigma,
        )

        # Remove BaseHead's generic output_layer
        if self.output_layer is not None:
            del self.output_layer
            self.output_layer = None

    def forward(
        self, x: Tensor, state: Optional[Dict[str, Any]] = None
    ) -> Tuple[Tensor, Dict[str, Any], Any]:
        x = self.process_input(x)

        # Value Stream
        v = _forward_hidden(self.value_hidden, self._value_activation, x)
        v = self.value_output(v)
        v = v.view(-1, 1, self.representation.num_features)

        # Advantage Stream
        a = _forward_hidden(self.advantage_hidden, self._advantage_activation, x)
        a = self.advantage_output(a)
        a = a.view(-1, self.num_actions, self.representation.num_features)

        # Aggregation: Q = V + (A - mean(A))
        a_mean = a.mean(dim=1, keepdim=True)
        q = v + a - a_mean

        new_state = state if state is not None else {}
        inference = self.representation.to_inference(q)
        return q, new_state, inference

    @property
    def semantic_type(self) -> Any:
        return ActionDistribution[self.get_structure()]
