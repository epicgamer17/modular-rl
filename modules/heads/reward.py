from typing import Tuple, Optional, Dict, Any, List
import torch
from torch import Tensor
from torch import nn
from .base import BaseHead, HeadOutput
from agents.learner.losses.representations import BaseRepresentation
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig
from configs.modules.heads.reward import ValuePrefixRewardHeadConfig
from modules.backbones.factory import BackboneFactory
from modules.backbones.mlp import build_dense, NoisyLinear


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
        name: Optional[str] = None,
        input_source: str = "default",
    ):
        super().__init__(
            arch_config,
            input_shape,
            representation,
            neck_config,
            name=name,
            input_source=input_source,
        )

        # 1. Heads now build their own feature architecture (neck)
        self.neck = BackboneFactory.create(neck_config, input_shape)
        self.output_shape = self.neck.output_shape
        self.flat_dim = self._get_flat_dim(self.output_shape)

        # 2. Heads now define their own Final Output layer
        self.output_layer = build_dense(
            in_features=self.flat_dim,
            out_features=self.representation.num_features,
            sigma=self.arch_config.noisy_sigma,
        )

    def reset_noise(self) -> None:
        """Propagate noise reset through the head's submodules."""
        if hasattr(self.neck, "reset_noise"):
            self.neck.reset_noise()
        if isinstance(self.output_layer, NoisyLinear):
            self.output_layer.reset_noise()

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
        is_inference: bool = False,
        **kwargs,
    ) -> HeadOutput:
        """Returns HeadOutput containing logits and projected reward scalar."""
        # 1. Processing neck -> flatten
        x = self.neck(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)

        # 2. Final Output Projection
        logits = self.output_layer(x)

        # 3. Mathematical Transform
        instant_reward = None
        if is_inference:
            instant_reward = self.representation.to_expected_value(logits)

        return HeadOutput(
            training_tensor=logits,
            inference_tensor=instant_reward,
            state=state if state is not None else {},
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
        name: Optional[str] = None,
        input_source: str = "default",
    ):
        # We call BaseHead init to avoid RewardHead's default output_layer logic
        BaseHead.__init__(
            self,
            arch_config,
            input_shape,
            representation,
            neck_config,
            name=name,
            input_source=input_source,
        )

        self.neck = BackboneFactory.create(neck_config, input_shape)
        self.output_shape = self.neck.output_shape
        self.flat_dim = self._get_flat_dim(self.output_shape)

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

    def reset_noise(self) -> None:
        if hasattr(self.neck, "reset_noise"):
            self.neck.reset_noise()
        if isinstance(self.output_layer, NoisyLinear):
            self.output_layer.reset_noise()

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
        is_inference: bool = False,
        **kwargs,
    ) -> HeadOutput:
        state = state if state is not None else {}

        # 1. Process neck -> flatten
        x = self.neck(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)

        # 2. Prepare for LSTM: (B, Seq=1, Features)
        x = x.unsqueeze(1)

        # Retrieve state using self.name prefix
        hidden = state.get(f"{self.name}_reward_hidden")
        step_count = state.get(f"{self.name}_step_count")
        parent_cumulative = state.get(
            f"{self.name}_cumulative_reward",
            torch.zeros(x.shape[0], 1, device=x.device),
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
            reset_mask = (step_count > 0) & (step_count % self.lstm_horizon_len == 0)

            if reset_mask.any():
                mask_idx = reset_mask.view(-1)
                h[:, mask_idx] = 0.0
                c[:, mask_idx] = 0.0
                hidden = (h, c)

                effective_parent_cumulative[reset_mask] = 0.0
                effective_step_count[reset_mask] = 0.0

        # LSTM step
        output, (h_n, c_n) = self.lstm(x, hidden)
        output = output.squeeze(1)

        # Final projection: Predicted Cumulative Reward (Value Prefix)
        logits = self.output_layer(output)
        # GET INSTANT REWARD
        instant_reward = None
        if is_inference:
            expected_cumulative = self.representation.to_expected_value(logits)
            if expected_cumulative.dim() == 1:
                expected_cumulative = expected_cumulative.unsqueeze(-1)
            instant_reward = (
                expected_cumulative - effective_parent_cumulative
            ).squeeze(-1)

        # New state
        new_state = state.copy()
        new_state.update(
            {
                f"{self.name}_reward_hidden": (h_n, c_n),
                f"{self.name}_step_count": effective_step_count + 1,
                f"{self.name}_cumulative_reward": expected_cumulative,
            }
        )

        return HeadOutput(
            training_tensor=logits,
            inference_tensor=instant_reward,
            state=new_state,
        )
