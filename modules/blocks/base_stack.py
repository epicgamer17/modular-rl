# modules/base_stacks.py (New File)
from typing import Callable, Tuple
import torch
from torch import nn, Tensor


class BaseStack(nn.Module):
    """Base class for sequential network stacks (Dense, Conv, Residual)."""

    def __init__(self, activation: nn.Module = nn.ReLU(), noisy_sigma: float = 0):
        super().__init__()
        self.activation = activation
        self.noisy = noisy_sigma != 0
        self._layers = nn.ModuleList()
        self._output_len = None

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        """Initializes all Conv2d and Linear layers in the stack."""

        def initialize_if_relevant(m: nn.Module):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # Only apply to weight, as bias might be handled by NoisyDense
                initializer(m.weight)

        self.apply(initialize_if_relevant)

    def reset_noise(self) -> None:
        """Resets the noise for all noisy layers."""
        if not self.noisy:
            return
        for layer in self._layers:
            if hasattr(layer, "reset_noise"):
                layer.reset_noise()

    def remove_noise(self) -> None:
        """Sets the noise to zero for all noisy layers (inference mode)."""
        if not self.noisy:
            return
        for layer in self._layers:
            if hasattr(layer, "remove_noise"):
                layer.remove_noise()

    @property
    def output_features(self):
        """Returns the final output size/channel count."""
        return self._output_len

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement forward method.")
