from typing import Tuple, Optional, Callable, Dict, Any
import torch
from torch import nn, Tensor
from agents.learner.losses.representations import BaseRepresentation
from modules.layers.noisy_linear import build_linear_layer


class BaseHead(nn.Module):
    """
    Base class for all network heads.
    Handles an optional neck (modular backbone) and standard initialization.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        neck: Optional[nn.Module] = None,
        noisy_sigma: float = 0.0,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.representation = representation

        # 1. Neck (optional modular backbone associated with the head)
        self.neck = neck if neck is not None else nn.Identity()

        # If neck has output_shape attribute, use it; otherwise infer it
        if hasattr(self.neck, "output_shape"):
            self.output_shape = self.neck.output_shape
        else:
            self.output_shape = input_shape

        self.flat_dim = self._get_flat_dim(self.output_shape)

        # 2. Final Output Layer
        self.output_layer = None
        if self.representation is not None:
            self.output_layer = build_linear_layer(
                in_features=self.flat_dim,
                out_features=self.representation.num_features,
                sigma=noisy_sigma,
            )

    def _get_flat_dim(self, shape: Tuple[int, ...]) -> int:
        flat = 1
        for dim in shape:
            flat *= dim
        return flat

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
