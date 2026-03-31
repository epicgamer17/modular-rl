from typing import Tuple
import torch
from torch import nn
from old_muzero.configs.modules.backbones.transformer import TransformerConfig


class TransformerBackbone(nn.Module):
    """Transformer backbone implementation."""

    def __init__(self, config: TransformerConfig, input_shape: Tuple[int, ...]):
        super().__init__()
        self.config = config
        self.input_shape = input_shape

        # input_shape: (Seq, Features) or (Features)
        if len(input_shape) == 1:
            input_dim = input_shape[0]
        else:
            input_dim = input_shape[-1]

        self.embedding = (
            nn.Linear(input_dim, config.d_model)
            if input_dim != config.d_model
            else nn.Identity()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="relu",  # Standard for Transformer
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        self.output_shape = (config.d_model,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) or (B, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)

        x = self.embedding(x)
        x = self.transformer_encoder(x)
        return x[:, -1, :]  # Return last token's representation

