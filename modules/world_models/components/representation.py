from typing import Tuple, Callable
import torch
from torch import nn
from configs.agents.muzero import MuZeroConfig
from modules.backbones.factory import BackboneFactory
from modules.utils import _normalize_hidden_state


class Representation(nn.Module):
    def __init__(self, config: MuZeroConfig, input_shape: Tuple[int]):
        super().__init__()
        assert (
            config.game.is_discrete
        ), "AlphaZero only works for discrete action space games (board games)"
        self.config = config
        self.net = BackboneFactory.create(config.representation_backbone, input_shape)
        self.output_shape = self.net.output_shape

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        self.net.initialize(initializer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        S = self.net(inputs)
        # Apply normalization to the final output of the representation network
        return _normalize_hidden_state(S)
