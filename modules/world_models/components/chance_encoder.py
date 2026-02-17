from typing import Tuple, Callable
import torch
from torch import nn
from configs.agents.muzero import MuZeroConfig
from modules.backbones.factory import BackboneFactory
from modules.utils import zero_weights_initializer


class ChanceEncoder(nn.Module):
    def __init__(
        self,
        config: MuZeroConfig,
        input_shape: Tuple[int, ...],
        num_codes: int = 32,
    ):
        """
        Args:
            config: MuZeroConfig containing chance_encoder_backbone.
            input_shape: tuple, e.g. (C, H, W) or (B, C, H, W).
            num_codes: embedding size output by encoder.
        """
        super().__init__()
        self.config = config
        self.num_codes = num_codes

        # Use modular backbone for Encoder
        self.net = BackboneFactory.create(config.chance_encoder_backbone, input_shape)

        # Output head: maps backbone output to num_codes
        backbone_output_shape = self.net.output_shape
        flat_dim = 1
        for d in backbone_output_shape:
            flat_dim *= d

        self.fc = nn.Linear(flat_dim, num_codes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            probs: (B, num_codes) - Softmax probabilities
            one_hot_st: (B, num_codes) - Straight-Through gradient flow
        """
        # 1. Processing to Logits
        x = self.net(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)
        x = self.fc(x)

        # 2. Softmax
        probs = x.softmax(dim=-1)

        # Convert to one-hot (B, num_codes)
        one_hot = torch.zeros_like(probs).scatter_(
            -1, torch.argmax(probs, dim=-1, keepdim=True), 1.0
        )

        # # 4. Straight-Through Estimator
        # # Forward: use one_hot
        # # Backward: use gradients of probs
        one_hot_st = (one_hot - probs).detach() + probs

        return probs, one_hot_st

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        self.net.initialize(initializer)
        zero_weights_initializer(self.fc)


# TODO ADD MORE CHANCE ENCODERS, gumbel softmax, output logits instead of probs, LightZero version, etc
