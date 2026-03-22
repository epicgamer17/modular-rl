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
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Recurrent forward pass: memory core always treats (B, T, D) sequence batches.
        Returns: (output_sequence, next_state_dict)
        """
        # (Batch, Time, Features) expected. Auto-flatten if (Batch, Time, C, H, W).
        if x.dim() > 3:
            # (B, T, C, H, W) -> (B, T, D)
            x = x.flatten(2, -1)

        assert x.dim() == 3, f"Memory core input must be (B, T, D), got shape {x.shape}"

        h = None
        if state is not None and "rnn_h" in state:
            if isinstance(self.rnn, nn.LSTM):
                h = (state["rnn_h"], state["rnn_c"])
            else:
                h = state["rnn_h"]

        output, h_n = self.rnn(x, h)
        
        next_state = {}
        if isinstance(self.rnn, nn.LSTM):
            next_state["rnn_h"] = h_n[0]
            next_state["rnn_c"] = h_n[1]
        else:
            next_state["rnn_h"] = h_n

        return output, next_state
