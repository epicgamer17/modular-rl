from typing import Tuple, Optional, Dict, Any
from torch import Tensor
from torch import nn
import torch
from .base import BaseHead
from modules.output_strategies import OutputStrategy
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig
from configs.modules.heads.reward import ValuePrefixRewardHeadConfig
from modules.dense import build_dense


class RewardHead(BaseHead):
    """
    Predicts the instant reward.
    Supports multiple output strategies (Regression, MuZero, C51, Dreamer).
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        strategy: Optional[OutputStrategy] = None,
        neck_config: Optional[BackboneConfig] = None,
    ):
        super().__init__(arch_config, input_shape, strategy, neck_config)

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
        return_scalar: bool = True,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        logits, new_state = super().forward(x, state)
        if return_scalar:
            return self.strategy.to_expected_value(logits), new_state
        return logits, new_state


class ValuePrefixRewardHead(RewardHead):
    """
    Predicts the instant reward using an LSTM to model value prefix.
    Encapsulates the recurrent logic and state.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        strategy: OutputStrategy,
        config: ValuePrefixRewardHeadConfig,
        neck_config: Optional[BackboneConfig] = None,
    ):
        # Pass strategy=None to avoid creating the default output layer in BaseHead
        super().__init__(
            arch_config, input_shape, strategy=None, neck_config=neck_config
        )
        self.strategy = strategy
        self.lstm_hidden_size = config.lstm_hidden_size
        self.lstm_horizon_len = config.lstm_horizon_len

        # LSTM input size is the output of the neck (flat_dim)
        self.lstm = nn.LSTM(
            input_size=self.flat_dim,
            hidden_size=self.lstm_hidden_size,
            batch_first=True,
        )

        # Output layer needs to map from LSTM hidden size to strategy bins
        self.output_layer = build_dense(
            in_features=self.lstm_hidden_size,
            out_features=self.strategy.num_bins,
            sigma=self.arch_config.noisy_sigma,
        )

    def get_initial_state(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, Any]:
        return {
            "reward_hidden": (
                torch.zeros(1, batch_size, self.lstm_hidden_size, device=device),
                torch.zeros(1, batch_size, self.lstm_hidden_size, device=device),
            )
        }

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
        return_scalar: bool = True,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        # Process neck
        x = self.process_input(x)  # (B, flat_dim)

        # Prepare for LSTM: (B, Seq=1, Features)
        x = x.unsqueeze(1)

        # Retrieve state
        hidden = state.get("reward_hidden") if state else None

        # LSTM step
        output, (h_n, c_n) = self.lstm(x, hidden)

        # Output is (B, 1, Hidden)
        output = output.squeeze(1)

        # New state
        new_state = {"reward_hidden": (h_n, c_n)}

        # Final projection
        logits = self.output_layer(output)

        if return_scalar:
            return self.strategy.to_expected_value(logits), new_state
        return logits, new_state
