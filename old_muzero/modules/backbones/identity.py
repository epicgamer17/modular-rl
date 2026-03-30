import torch.nn as nn
from torch import Tensor
from typing import Tuple


class IdentityBackbone(nn.Module):
    """
    A pass-through backbone that returns the input as is.
    """

    def __init__(self, config, input_shape: Tuple[int]):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = input_shape  # Output shape is same as input

    def forward(self, x: Tensor) -> Tensor:
        return x

    def get_output_shape(self) -> Tuple[int]:
        return self.output_shape

    def reset_noise(self):
        pass  # pragma: no cover
