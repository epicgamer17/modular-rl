from typing import Tuple
import torch
from torch import nn
from modules.blocks.dense import DenseStack


class DenseBackbone(nn.Module):
    """Dense (MLP) backbone implementation using DenseStack."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        widths: list[int],
        activation: str = "relu",
        noisy_sigma: float = 0.0,
        norm_type: str = "none",
    ):
        super().__init__()
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
            widths=widths,
            activation=activation,
            noisy_sigma=noisy_sigma,
            norm_type=norm_type,
        )

        self.output_shape = (self.stack.output_width,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.flatten(1, -1)
        return self.stack(x)

    def reset_noise(self) -> None:
        self.stack.reset_noise()
