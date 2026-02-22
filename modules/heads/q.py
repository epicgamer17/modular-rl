from typing import Tuple, Optional, Callable
import torch
from torch import nn, Tensor

from .base import BaseHead
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig
from modules.heads.strategies import OutputStrategy
from modules.blocks.dense import DenseStack, build_dense


class QHead(BaseHead):
    """
    Standard Q-Network Head.
    Structure: [Neck (Optional)] -> [Hidden Layers] -> [Output Layer] -> [Strategy (Reshape)]
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        strategy: OutputStrategy,
        hidden_widths: list[int],
        num_actions: int,
        neck_config: Optional[BackboneConfig] = None,
    ):
        # Initialize BaseHead (handles neck)
        # Note: BaseHead creates output_layer based on flat_dim.
        # We need to intercept this because we insert hidden layers between neck and output.
        # But BaseHead architecture is: Neck -> Output.
        # We can implement hidden layers as a "Neck" via DenseBackbone?
        # No, BaseHead is designed for [Pre-Head Backbone] -> [Head Output].
        # If we want hidden layers strictly *inside* the head (post-neck), we should manage them here.

        super().__init__(arch_config, input_shape, strategy, neck_config)

        self.num_actions = num_actions
        self.input_dim = self.flat_dim  # From super()

        # 1. Hidden Layers
        self.hidden_layers = DenseStack(
            initial_width=self.input_dim,
            widths=hidden_widths,
            activation=self.arch_config.activation,
            noisy_sigma=self.arch_config.noisy_sigma,
            norm_type=self.arch_config.norm_type,
        )

        # 2. Output Layer (Overwrites BaseHead's output_layer)
        # Output size: num_actions * atoms
        self.output_layer = build_dense(
            in_features=self.hidden_layers.output_width,
            out_features=self.strategy.num_bins * self.num_actions,
            sigma=self.arch_config.noisy_sigma,
        )

    def initialize(
        self, initializer: Optional[Callable[[Tensor], None]] = None
    ) -> None:
        super().initialize(initializer)  # Inits neck and output_layer
        self.hidden_layers.initialize(initializer)

    def reset_noise(self) -> None:
        super().reset_noise()
        self.hidden_layers.reset_noise()

    def forward(self, x: Tensor) -> Tensor:
        # 1. Neck + Flatten
        x = self.process_input(x)

        # 2. Hidden Layers
        x = self.hidden_layers(x)

        # 3. Output Layer
        logits = self.output_layer(x)

        # 4. Reshape for actions (B, actions, atoms) if needed by strategy?
        # Strategies like C51 expect (B, actions, atoms)
        # Strategies like ScalarStrategy expect (B, actions)

        if self.strategy.num_bins > 1:
            logits = logits.view(-1, self.num_actions, self.strategy.num_bins)

        return logits


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
        strategy: OutputStrategy,
        value_hidden_widths: list[int],
        advantage_hidden_widths: list[int],
        num_actions: int,
        neck_config: Optional[BackboneConfig] = None,
    ):
        super().__init__(arch_config, input_shape, strategy, neck_config)

        self.num_actions = num_actions
        self.input_dim = self.flat_dim

        # 1. Value Stream
        self.value_hidden = DenseStack(
            initial_width=self.input_dim,
            widths=value_hidden_widths,
            activation=self.arch_config.activation,
            noisy_sigma=self.arch_config.noisy_sigma,
            norm_type=self.arch_config.norm_type,
        )
        self.value_output = build_dense(
            in_features=self.value_hidden.output_width,
            out_features=self.strategy.num_bins,  # 1 value * atoms
            sigma=self.arch_config.noisy_sigma,
        )

        # 2. Advantage Stream
        self.advantage_hidden = DenseStack(
            initial_width=self.input_dim,
            widths=advantage_hidden_widths,
            activation=self.arch_config.activation,
            noisy_sigma=self.arch_config.noisy_sigma,
            norm_type=self.arch_config.norm_type,
        )
        self.advantage_output = build_dense(
            in_features=self.advantage_hidden.output_width,
            out_features=self.strategy.num_bins * self.num_actions,  # N actions * atoms
            sigma=self.arch_config.noisy_sigma,
        )

        # Remove BaseHead's generic output_layer as we have specialized ones
        if self.output_layer is not None:
            del self.output_layer
            self.output_layer = None

    def initialize(
        self, initializer: Optional[Callable[[Tensor], None]] = None
    ) -> None:
        # Init neck
        super().initialize(initializer)

        # Init streams
        init_fn = initializer or self.arch_config.kernel_initializer
        out_init_fn = (
            initializer
            or self.arch_config.output_layer_initializer
            or self.arch_config.kernel_initializer
        )

        self.value_hidden.initialize(init_fn)
        self.advantage_hidden.initialize(init_fn)

        if hasattr(self.value_output, "initialize"):
            self.value_output.initialize(out_init_fn)
        elif out_init_fn:
            self.value_output.apply(out_init_fn)

        if hasattr(self.advantage_output, "initialize"):
            self.advantage_output.initialize(out_init_fn)
        elif out_init_fn:
            self.advantage_output.apply(out_init_fn)

    def reset_noise(self) -> None:
        super().reset_noise()  # Neck
        self.value_hidden.reset_noise()
        self.advantage_hidden.reset_noise()
        if hasattr(self.value_output, "reset_noise"):
            self.value_output.reset_noise()
        if hasattr(self.advantage_output, "reset_noise"):
            self.advantage_output.reset_noise()

    def forward(self, x: Tensor) -> Tensor:
        # Neck
        x = self.process_input(x)

        # Value Stream
        v = self.value_hidden(x)
        v = self.value_output(v)  # (B, atoms)
        v = v.view(-1, 1, self.strategy.num_bins)  # (B, 1, atoms)

        # Advantage Stream
        a = self.advantage_hidden(x)
        a = self.advantage_output(a)  # (B, actions * atoms)
        a = a.view(-1, self.num_actions, self.strategy.num_bins)  # (B, actions, atoms)

        # Aggregation: Q = V + (A - mean(A))
        a_mean = a.mean(dim=1, keepdim=True)
        q = v + a - a_mean

        # Output is (B, actions, atoms)
        # If atoms=1, we might want (B, actions) but for consistency let's keep dimensions until final squeeze?
        # RainbowNetwork currently expects:
        # if atom_size == 1: squeeze(-1)
        # elze: softmax

        return q
