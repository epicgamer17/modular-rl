from typing import Tuple, Optional
import torch
from torch import nn


class RecurrentBackbone(nn.Module):
    """GRU/LSTM backbone implementation."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        rnn_type: str = "gru",
        hidden_size: int = 256,
        num_layers: int = 1,
    ):
        super().__init__()
        self.input_shape = input_shape

        # Input shape (Seq, Features) or (Features)
        if len(input_shape) == 1:
            input_dim = input_shape[0]
        else:
            input_dim = input_shape[-1]

        if rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )

        self.output_shape = (hidden_size,)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, D) or (B, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)

        output, h_n = self.rnn(x, h)
        return output[:, -1, :], h_n  # Return last output and hidden state
