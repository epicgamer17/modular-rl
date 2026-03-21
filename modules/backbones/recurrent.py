from typing import Tuple, Optional, Union, Dict, Any
import torch
from torch import nn
from configs.modules.backbones.recurrent import RecurrentConfig


class RecurrentBackbone(nn.Module):
    """GRU/LSTM backbone implementation."""

    def __init__(self, config: RecurrentConfig, input_shape: Tuple[int, ...]):
        super().__init__()
        self.config = config
        self.input_shape = input_shape

        # Input shape (Seq, Features) or (Features)
        if len(input_shape) == 1:
            input_dim = input_shape[0]
        else:
            input_dim = input_shape[-1]

        if config.rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
            )

        self.output_shape = (config.hidden_size,)

    def forward(
        self, x: torch.Tensor, h: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Recurrent forward pass: memory core always treats (B, T, D) sequence batches.
        Returns: (output_sequence, last_hidden_state)
        """
        # --- STRICT MEMORY CONTRACT ---
        # Memory cores always process (Batch, Time, Features)
        assert x.dim() == 3, f"Memory core input must be (B, T, D), got shape {x.shape}"

        output, h_n = self.rnn(x, h)
        return output, h_n

