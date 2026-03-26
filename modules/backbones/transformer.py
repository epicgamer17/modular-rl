from typing import Tuple
import torch
from torch import nn
from configs.modules.backbones.transformer import TransformerConfig


class TransformerBackbone(nn.Module):
    """Transformer backbone implementation."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.input_shape = input_shape

        # input_shape: (Seq, Features) or (Features)
        if len(input_shape) == 1:
            input_dim = input_shape[0]
        else:
            input_dim = input_shape[-1]

        self.embedding = (
            nn.Linear(input_dim, d_model)
            if input_dim != d_model
            else nn.Identity()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",  # Standard for Transformer
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.output_shape = (d_model,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) or (B, D) or (B, L, C, H, W)
        if x.dim() > 3:
            # (B, L, C, H, W) -> (B, L, Features)
            x = x.flatten(2, -1)

        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)

        x = self.embedding(x)
        x = self.transformer_encoder(x)
        return x[:, -1, :]  # Return last token's representation

