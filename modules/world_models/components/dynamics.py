from typing import Tuple, Callable, Optional, Dict, Any
import torch
from torch import nn
import torch.nn.functional as F

from configs.agents.muzero import MuZeroConfig
from modules.embeddings.action_embedding import ActionEncoder
from modules.backbones.factory import BackboneFactory
from modules.utils import _normalize_hidden_state


class BaseDynamics(nn.Module):
    """Base class for Dynamics and AfterstateDynamics, handling action fusion and core block."""

    def __init__(
        self,
        config: MuZeroConfig,
        input_shape: Tuple[int],
        num_actions: int,
        action_embedding_dim: int,
        layer_prefix: str,
    ):
        super().__init__()
        self.config = config
        self.action_embedding_dim = action_embedding_dim
        is_continuous = not self.config.game.is_discrete

        # 1. Action Encoder
        self.action_encoder = ActionEncoder(
            action_space_size=num_actions,
            embedding_dim=self.action_embedding_dim,
            is_continuous=is_continuous,
            single_action_plane=(layer_prefix == "dynamics"),
        )

        # 2. Fusion Layer (Move from ActionEncoder to Dynamics)
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
        if layer_prefix == "dynamics":
            bb_cfg = config.dynamics_backbone
        elif layer_prefix == "afterstate_dynamics":
            bb_cfg = config.afterstate_dynamics_backbone

        self.net = BackboneFactory.create(bb_cfg, input_shape)
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
        config: MuZeroConfig,
        input_shape: Tuple[int],
        num_actions: int,
        action_embedding_dim: int,
    ):
        # dynamics layers uses the "dynamics" prefix
        super().__init__(
            config, input_shape, num_actions, action_embedding_dim, "dynamics"
        )
        # NOTE: RewardHead and ToPlayHead are GONE from here!

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
        config: MuZeroConfig,
        input_shape: Tuple[int],
        num_actions: int,
        action_embedding_dim: int,
    ):
        super().__init__(
            config,
            input_shape,
            num_actions,
            action_embedding_dim,
            "afterstate_dynamics",
        )

    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # The base class handles fusion and processing, returning the normalized hidden state (afterstate)
        afterstate = self._fuse_and_process(hidden_state, action)
        return afterstate
