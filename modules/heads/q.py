from typing import Tuple, Optional, Callable, Dict, Any, List
import torch
from torch import nn, Tensor

from .base import BaseHead, HeadOutput
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig
from agents.learner.losses.representations import BaseRepresentation
from modules.backbones.mlp import MLPBackbone, build_dense, NoisyLinear
from modules.utils import build_normalization_layer


class QHead(BaseHead):
    """
    Standard Q-Network Head.
    Structure: [Neck (Optional)] -> [Hidden Layers] -> [Output Layer] -> [Strategy (Reshape)]
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        hidden_widths: List[int],
        num_actions: int,
        neck_config: Optional[BackboneConfig] = None,
    ):
        super().__init__(arch_config, input_shape, representation, neck_config)

        self.num_actions = num_actions
        self.input_dim = self.flat_dim  # From super()
        self.noisy = self.arch_config.noisy_sigma != 0

        # 1. Hidden Layers constructed directly via nn.Sequential
        layers = []
        current_width = self.input_dim
        for width in hidden_widths:
            use_bias = self.arch_config.norm_type == "none"
            layers.append(
                build_dense(
                    current_width,
                    width,
                    sigma=self.arch_config.noisy_sigma,
                    bias=use_bias,
                )
            )

            if self.arch_config.norm_type != "none":
                layers.append(
                    build_normalization_layer(self.arch_config.norm_type, width, dim=1)
                )

            layers.append(self.arch_config.activation)
            current_width = width

        self.hidden_layers = nn.Sequential(*layers)

        # 2. Output Layer
        self.output_layer = build_dense(
            in_features=current_width,
            out_features=self.representation.num_features * self.num_actions,
            sigma=self.arch_config.noisy_sigma,
        )

    def reset_noise(self) -> None:
        super().reset_noise()
        if not self.noisy:
            return
        for m in self.hidden_layers.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
        if isinstance(self.output_layer, NoisyLinear):
            self.output_layer.reset_noise()

    def forward(self, x: Tensor, state: Optional[Dict[str, Any]] = None) -> HeadOutput:
        # 1. Neck + Flatten
        x = self.process_input(x)

        # 2. Hidden Layers
        x = self.hidden_layers(x)

        # 3. Output Layer
        logits = self.output_layer(x)
        logits = logits.view(-1, self.num_actions, self.representation.num_features)

        # 4. Standard Return
        new_state = state if state is not None else {}
        inference = self.representation.to_inference(logits)

        return HeadOutput(
            training_tensor=logits,
            inference_tensor=inference,
            state=new_state,
        )


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
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        value_hidden_widths: List[int],
        advantage_hidden_widths: List[int],
        num_actions: int,
        neck_config: Optional[BackboneConfig] = None,
    ):
        super().__init__(arch_config, input_shape, representation, neck_config)

        self.num_actions = num_actions
        self.input_dim = self.flat_dim
        self.noisy = self.arch_config.noisy_sigma != 0

        # Helper to build a hidden stream
        def build_stream(widths: List[int]):
            layers = []
            curr_w = self.input_dim
            for w in widths:
                use_bias = self.arch_config.norm_type == "none"
                layers.append(
                    build_dense(
                        curr_w, w, sigma=self.arch_config.noisy_sigma, bias=use_bias
                    )
                )
                if self.arch_config.norm_type != "none":
                    layers.append(
                        build_normalization_layer(self.arch_config.norm_type, w, dim=1)
                    )
                layers.append(self.arch_config.activation)
                curr_w = w
            return nn.Sequential(*layers), curr_w

        # 1. Value Stream
        self.value_hidden, v_width = build_stream(value_hidden_widths)
        self.value_output = build_dense(
            in_features=v_width,
            out_features=self.representation.num_features,
            sigma=self.arch_config.noisy_sigma,
        )

        # 2. Advantage Stream
        self.advantage_hidden, a_width = build_stream(advantage_hidden_widths)
        self.advantage_output = build_dense(
            in_features=a_width,
            out_features=self.representation.num_features * self.num_actions,
            sigma=self.arch_config.noisy_sigma,
        )

        # Remove BaseHead's generic output_layer
        if self.output_layer is not None:
            del self.output_layer
            self.output_layer = None

    def reset_noise(self) -> None:
        super().reset_noise()
        if not self.noisy:
            return

        for stream in [self.value_hidden, self.advantage_hidden]:
            for m in stream.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()

        for out in [self.value_output, self.advantage_output]:
            if isinstance(out, NoisyLinear):
                out.reset_noise()

    def forward(self, x: Tensor, state: Optional[Dict[str, Any]] = None) -> HeadOutput:
        # Neck
        x = self.process_input(x)

        # Value Stream
        v = self.value_hidden(x)
        v = self.value_output(v)  # (B, atoms)
        v = v.view(-1, 1, self.representation.num_features)  # (B, 1, atoms)

        # Advantage Stream
        a = self.advantage_hidden(x)
        a = self.advantage_output(a)  # (B, actions * atoms)
        a = a.view(-1, self.num_actions, self.representation.num_features)

        # Aggregation: Q = V + (A - mean(A))
        a_mean = a.mean(dim=1, keepdim=True)
        q = v + a - a_mean

        new_state = state if state is not None else {}
        inference = self.representation.to_inference(q)

        return HeadOutput(
            training_tensor=q,
            inference_tensor=inference,
            state=new_state,
        )
