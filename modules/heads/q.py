from typing import Tuple, Optional, Callable, Dict, Any
import torch
from torch import nn, Tensor

from .base import BaseHead
from agents.learner.losses.representations import BaseRepresentation
from modules.blocks.linear import LinearStack, build_linear_block


class QHead(BaseHead):
    """
    Standard Q-Network Head.
    Structure: [Neck (Optional)] -> [Hidden Layers] -> [Output Layer] -> [Strategy (Reshape)]
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        hidden_widths: list[int],
        num_actions: int,
        neck: Optional[nn.Module] = None,
        noisy_sigma: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        norm_type: str = "none",
    ):
        super().__init__(input_shape, representation, neck, noisy_sigma)

        self.num_actions = num_actions
        self.input_dim = self.flat_dim  # From super()

        # 1. Hidden Layers
        self.hidden_layers = LinearStack(
            initial_width=self.input_dim,
            widths=hidden_widths,
            activation=activation,
            noisy_sigma=noisy_sigma,
            norm_type=norm_type,
        )

        # 2. Output Layer (Overwrites BaseHead's output_layer)
        # Output size: num_actions * atoms
        self.output_layer = build_linear_block(
            in_features=self.hidden_layers.output_width,
            out_features=self.representation.num_features * self.num_actions,
            sigma=noisy_sigma,
        )

    def reset_noise(self) -> None:
        super().reset_noise()
        self.hidden_layers.reset_noise()

    def forward(
        self, x: Tensor, state: Optional[Dict[str, Any]] = None
    ) -> Tuple[Tensor, Dict[str, Any], Any]:
        # 1. Neck + Flatten
        x = self.process_input(x)

        # 2. Hidden Layers
        x = self.hidden_layers(x)

        # 3. Output Layer
        logits = self.output_layer(x)
        logits = logits.view(-1, self.num_actions, self.representation.num_features)

        # 4. Standard Return: (logits, state, inference)
        new_state = state if state is not None else {}
        inference = self.representation.to_inference(logits)

        return logits, new_state, inference


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
        value_hidden_widths: list[int],
        advantage_hidden_widths: list[int],
        num_actions: int,
        neck: Optional[nn.Module] = None,
        noisy_sigma: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        norm_type: str = "none",
    ):
        super().__init__(input_shape, representation, neck, noisy_sigma)

        self.num_actions = num_actions
        self.input_dim = self.flat_dim

        # 1. Value Stream
        self.value_hidden = LinearStack(
            initial_width=self.input_dim,
            widths=value_hidden_widths,
            activation=activation,
            noisy_sigma=noisy_sigma,
            norm_type=norm_type,
        )
        self.value_output = build_linear_block(
            in_features=self.value_hidden.output_width,
            out_features=self.representation.num_features,  # 1 value * atoms
            sigma=noisy_sigma,
        )

        # 2. Advantage Stream
        self.advantage_hidden = LinearStack(
            initial_width=self.input_dim,
            widths=advantage_hidden_widths,
            activation=activation,
            noisy_sigma=noisy_sigma,
            norm_type=norm_type,
        )
        self.advantage_output = build_linear_block(
            in_features=self.advantage_hidden.output_width,
            out_features=self.representation.num_features
            * self.num_actions,  # N actions * atoms
            sigma=noisy_sigma,
        )

        # Remove BaseHead's generic output_layer as we have specialized ones
        if self.output_layer is not None:
            del self.output_layer
            self.output_layer = None

    def reset_noise(self) -> None:
        super().reset_noise()  # Neck
        self.value_hidden.reset_noise()
        self.advantage_hidden.reset_noise()
        if hasattr(self.value_output, "reset_noise"):
            self.value_output.reset_noise()
        if hasattr(self.advantage_output, "reset_noise"):
            self.advantage_output.reset_noise()

    def forward(
        self, x: Tensor, state: Optional[Dict[str, Any]] = None
    ) -> Tuple[Tensor, Dict[str, Any], Any]:
        # Neck
        x = self.process_input(x)

        # Value Stream
        v = self.value_hidden(x)
        v = self.value_output(v)  # (B, atoms)
        v = v.view(-1, 1, self.representation.num_features)  # (B, 1, atoms)

        # Advantage Stream
        a = self.advantage_hidden(x)
        a = self.advantage_output(a)  # (B, actions * atoms)
        a = a.view(
            -1, self.num_actions, self.representation.num_features
        )  # (B, actions, atoms)

        # Aggregation: Q = V + (A - mean(A))
        a_mean = a.mean(dim=1, keepdim=True)
        q = v + a - a_mean

        new_state = state if state is not None else {}
        inference = self.representation.to_inference(q)

        return q, new_state, inference
