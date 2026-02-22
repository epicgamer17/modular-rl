from typing import Tuple, Optional, Callable, Dict, Any
import torch
from torch import nn, Tensor
from modules.backbones.factory import BackboneFactory
from configs.modules.backbones.base import BackboneConfig
from configs.modules.architecture_config import ArchitectureConfig
from modules.heads.strategies import OutputStrategy
from modules.blocks.dense import build_dense


class BaseHead(nn.Module):
    """
    Base class for all network heads.
    Handles an optional neck (modular backbone) and standard initialization.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        strategy: Optional[OutputStrategy] = None,
        neck_config: Optional[BackboneConfig] = None,
    ):
        super().__init__()
        self.arch_config = arch_config
        self.input_shape = input_shape
        self.strategy = strategy

        # 1. Neck (optional modular backbone associated with the head)
        self.neck = BackboneFactory.create(neck_config, input_shape)
        self.output_shape = self.neck.output_shape
        self.flat_dim = self._get_flat_dim(self.output_shape)

        # 2. Final Output Layer
        self.output_layer = None
        if self.strategy is not None:
            self.output_layer = build_dense(
                in_features=self.flat_dim,
                out_features=self.strategy.num_bins,
                sigma=self.arch_config.noisy_sigma,
            )

    def _get_flat_dim(self, shape: Tuple[int, ...]) -> int:
        flat = 1
        for dim in shape:
            flat *= dim
        return flat

    def initialize(
        self, initializer: Optional[Callable[[Tensor], None]] = None
    ) -> None:
        """Initializes the neck and the output layer using the architecture config or provided initializer."""
        # Initialize neck
        if hasattr(self.neck, "initialize"):
            # Prefer provided initializer, then config
            init_fn = initializer or self.arch_config.kernel_initializer
            if init_fn:
                self.neck.initialize(init_fn)

        # Initialize the final output layer (defined in subclasses)
        if self.output_layer is not None:
            init_fn = (
                initializer
                or self.arch_config.output_layer_initializer
                or self.arch_config.kernel_initializer
            )
            if hasattr(self.output_layer, "initialize"):
                self.output_layer.initialize(init_fn)
            elif init_fn:
                self.output_layer.apply(init_fn)

    def reset_noise(self) -> None:
        if hasattr(self.neck, "reset_noise"):
            self.neck.reset_noise()
        if self.output_layer is not None and hasattr(self.output_layer, "reset_noise"):
            self.output_layer.reset_noise()

    def forward(
        self, x: Tensor, state: Optional[Dict[str, Any]] = None
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Standard forward pass: neck -> output_layer -> strategy."""
        x = self.process_input(x)
        logits = self.output_layer(x)
        return logits, state if state is not None else {}

    def process_input(self, x: Tensor) -> Tensor:
        """Helper to pass input through neck and flatten it."""
        x = self.neck(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)
        return x
