from typing import Tuple, Callable, Optional, Dict, Any
import torch
from torch import nn
import torch.nn.functional as F

from modules.embeddings.action_embedding import ActionEncoder
from modules.utils import _normalize_hidden_state


class BaseDynamics(nn.Module):
    """Base class for Dynamics and AfterstateDynamics, handling action fusion and core block."""

    def __init__(
        self,
        backbone: nn.Module,
        action_encoder: nn.Module,
        input_shape: Tuple[int, ...],
        action_embedding_dim: int,
    ):
        super().__init__()
        self.action_embedding_dim = action_embedding_dim

        # 1. Action Encoder (Pass in prepared encoder)
        self.action_encoder = action_encoder

        # 2. Fusion Layer
        if len(input_shape) == 3:
            # Image input (C, H, W)
            self.num_channels = input_shape[0]
            in_channels = self.num_channels + self.action_embedding_dim
            out_channels = self.num_channels
            self.fusion = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            )
            self.fusion_bn = nn.BatchNorm2d(out_channels)
        else:
            # Vector input (C,) or (D,)
            self.input_size = input_shape[0]
            in_features = self.input_size + self.action_embedding_dim
            out_features = self.input_size
            self.fusion = nn.Linear(in_features, out_features, bias=False)
            self.fusion_bn = nn.BatchNorm1d(out_features)

        # 3. Core Network Block
        self.net = backbone
        self.output_shape = self.net.output_shape

    def _fuse_and_process(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        # Embed action
        action_embedded = self.action_encoder(action, hidden_state.shape)

        # Concatenate and fuse
        x = torch.cat((hidden_state, action_embedded), dim=1)
        x = self.fusion(x)
        # x = self.fusion_bn(x) # BN is often omitted or placed after ReLU in some MuZero implementations

        # Residual Connection
        x = x + hidden_state
        S = F.relu(x)

        # Process through the main network block
        S = self.net(S)

        # Apply normalization to the final output of the dynamics network
        next_hidden_state = _normalize_hidden_state(S)

        return next_hidden_state


class Dynamics(BaseDynamics):
    def __init__(
        self,
        backbone: nn.Module,
        action_encoder: nn.Module,
        input_shape: Tuple[int, ...],
        action_embedding_dim: int,
    ):
        super().__init__(backbone, action_encoder, input_shape, action_embedding_dim)

    def forward(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Embed, fuse, residual, backbone, normalize
        next_hidden_state = self._fuse_and_process(hidden_state, action)

        # 2. Return ONLY the physical consequence
        return next_hidden_state


class AfterstateDynamics(BaseDynamics):
    def __init__(
        self,
        backbone: nn.Module,
        action_encoder: nn.Module,
        input_shape: Tuple[int, ...],
        action_embedding_dim: int,
    ):
        super().__init__(backbone, action_encoder, input_shape, action_embedding_dim)

    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # The base class handles fusion and processing, returning the normalized hidden state (afterstate)
        afterstate = self._fuse_and_process(hidden_state, action)
        return afterstate
