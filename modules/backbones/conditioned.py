from typing import Tuple, Optional, Dict, Any
import torch
from torch import nn
import torch.nn.functional as F

from modules.embeddings.action_embedding import ActionEncoder
from modules.backbones.factory import BackboneFactory
from modules.utils import _normalize_hidden_state

class ConditionedBackbone(nn.Module):
    """
    A backbone that fuses a hidden state with an external condition (like an action).
    Used primarily for Dynamics and AfterstateDynamics.
    """

    def __init__(
        self,
        config: Any,
        input_shape: Tuple[int, ...],
        num_actions: int,
        action_embedding_dim: int,
        backbone_config: Any,
    ):
        super().__init__()
        self.config = config
        self.action_embedding_dim = action_embedding_dim

        # 1. Action Encoder
        self.action_encoder = ActionEncoder(
            action_space_size=num_actions,
            embedding_dim=self.action_embedding_dim,
        )

        # 2. Fusion Layer
        if len(input_shape) == 3:
            # Image input (C, H, W)
            self.num_channels = input_shape[0]
            in_channels = self.num_channels + self.action_embedding_dim
            out_channels = self.num_channels
            self.fusion = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            )
        else:
            # Vector input (C,) or (D,)
            self.input_size = input_shape[0]
            in_features = self.input_size + self.action_embedding_dim
            out_features = self.input_size
            self.fusion = nn.Linear(in_features, out_features, bias=False)

        # 3. Core Network Block
        self.net = BackboneFactory.create(backbone_config, input_shape)
        self.num_actions = num_actions
        self.output_shape = self.net.output_shape

    def forward(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        # Handle index-based discrete actions (e.g. from MCTS)
        if action.dtype in [torch.long, torch.int] or (action.dim() == 1 and action.shape[0] == hidden_state.shape[0]):
            if action.dim() > 1 and action.shape[-1] == 1:
                action = action.squeeze(-1)
            if action.dim() == 1:
                action = F.one_hot(action.long(), num_classes=self.num_actions).float()
        
        # Ensure correct device
        action = action.to(hidden_state.device)

        # Embed action
        action_embedded = self.action_encoder(action, hidden_state.shape)

        # Concatenate and fuse
        x = torch.cat((hidden_state, action_embedded), dim=1)
        x = self.fusion(x)

        # Residual Connection
        x = x + hidden_state
        S = F.relu(x)

        # Process through the main network block
        S = self.net(S)

        # Apply normalization to the final output
        next_hidden_state = _normalize_hidden_state(S)

        return next_hidden_state

    def reset_noise(self) -> None:
        if hasattr(self.net, "reset_noise"):
            self.net.reset_noise()
