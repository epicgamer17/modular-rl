from typing import Tuple, Optional, Dict, Any, List
from torch import Tensor
from torch import nn
import torch
from .base import BaseHead, HeadOutput
from agents.learner.losses.representations import BaseRepresentation
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig
from configs.modules.heads.reward import ValuePrefixRewardHeadConfig
from modules.backbones.dense import build_dense


class RewardHead(BaseHead):
    """
    Predicts the instant reward.
    Supports multiple output strategies (Scalar, MuZero, C51, Dreamer).
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        neck_config: Optional[BackboneConfig] = None,
    ):
        super().__init__(arch_config, input_shape, representation, neck_config)

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> HeadOutput:
        """Returns HeadOutput with (logits, instant_reward, state)"""
        head_out = super().forward(x, state)
        instant_reward = self.representation.to_expected_value(head_out.training_tensor)
        return HeadOutput(
            training_tensor=head_out.training_tensor,
            inference_tensor=instant_reward,
            state=head_out.state,
        )


class ValuePrefixRewardHead(RewardHead):
    """
    Predicts the instant reward using an LSTM to model value prefix.
    Encapsulates the recurrent logic and state.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        config: ValuePrefixRewardHeadConfig,
        neck_config: Optional[BackboneConfig] = None,
    ):
        # Pass representation=None to avoid creating the default output layer in BaseHead
        super().__init__(
            arch_config, input_shape, representation=None, neck_config=neck_config
        )
        self.representation = representation
        self.lstm_hidden_size = config.lstm_hidden_size
        self.lstm_horizon_len = config.lstm_horizon_len

        # LSTM input size is the output of the neck (flat_dim)
        self.lstm = nn.LSTM(
            input_size=self.flat_dim,
            hidden_size=self.lstm_hidden_size,
            batch_first=True,
        )

        # Output layer needs to map from LSTM hidden size to representation features
        self.output_layer = build_dense(
            in_features=self.lstm_hidden_size,
            out_features=self.representation.num_features,
            sigma=self.arch_config.noisy_sigma,
        )

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> HeadOutput:
        state = state if state is not None else {}

        # Process neck
        x = self.process_input(x)  # (B, flat_dim)

        # Prepare for LSTM: (B, Seq=1, Features)
        x = x.unsqueeze(1)

        # Retrieve state
        hidden = state.get("reward_hidden")
        step_count = state.get("step_count")
        parent_cumulative = state.get(
            "cumulative_reward", torch.zeros(x.shape[0], 1, device=x.device)
        )

        if hidden is None:
            hidden = (
                torch.zeros(1, x.shape[0], self.lstm_hidden_size, device=x.device),
                torch.zeros(1, x.shape[0], self.lstm_hidden_size, device=x.device),
            )

        if step_count is None:
            batch_size = x.shape[0]
            step_count = torch.zeros(batch_size, 1, device=x.device)

        # Horizon Logic: Reset hidden state for elements where horizon is reached
        effective_parent_cumulative = parent_cumulative.clone()
        effective_step_count = step_count.clone()

        if hidden is not None:
            h, c = hidden
            # Identify indices that need reset
            # step_count > 0 and step_count % horizon == 0
            reset_mask = (step_count > 0) & (step_count % self.lstm_horizon_len == 0)

            if reset_mask.any():
                # Reset corresponding batch elements in h and c
                mask_idx = reset_mask.view(-1)
                h[:, mask_idx] = 0.0
                c[:, mask_idx] = 0.0
                hidden = (h, c)

                # USER FIX: Reset effective parent and step count for subtraction/accumulation
                effective_parent_cumulative[reset_mask] = 0.0
                effective_step_count[reset_mask] = 0.0

        # LSTM step
        output, (h_n, c_n) = self.lstm(x, hidden)

        # Output is (B, 1, Hidden)
        output = output.squeeze(1)

        # Final projection: Predicted Cumulative Reward (Value Prefix)
        logits = self.output_layer(output)
        expected_cumulative = self.representation.to_expected_value(logits)
        # Ensure (B, 1) shape for consistent accumulation with parent_cumulative
        if expected_cumulative.dim() == 1:
            expected_cumulative = expected_cumulative.unsqueeze(-1)

        # GET INSTANT REWARD: subtract (potentially reset) parent cumulative → (B,)
        instant_reward = (expected_cumulative - effective_parent_cumulative).squeeze(-1)

        # New state: Preserve all existing keys and update internal ones
        new_state = state.copy()
        new_state.update(
            {
                "reward_hidden": (h_n, c_n),
                "step_count": effective_step_count + 1,
                "cumulative_reward": expected_cumulative,
            }
        )

        return HeadOutput(
            training_tensor=logits,
            inference_tensor=instant_reward,
            state=new_state,
        )
