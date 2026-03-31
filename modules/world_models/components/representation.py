from typing import Tuple, Callable
import torch
from torch import nn
from modules.utils import _normalize_hidden_state


class Representation(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.net = backbone
        self.input_shape = getattr(backbone, "input_shape", None)
        self.output_shape = backbone.output_shape

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        S = self.net(inputs)
        # Apply normalization to the final output of the representation network
        return _normalize_hidden_state(S)
