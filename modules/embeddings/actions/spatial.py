import torch
from torch import nn
from torch import Tensor
from typing import Tuple


class SpatialActionEmbedding(nn.Module):
    """
    Encodes actions as a 1.0 at the (x, y) coordinate on a spatial grid.
    Used for board games like Tic-Tac-Toe, Go, Chess where actions map to board positions.
    """
    def __init__(
        self,
        num_actions: int,
        h: int,
        w: int,
        embedding_dim: int = 1,
        single_action_plane: bool = False,
    ):
        super().__init__()
        assert num_actions == h * w, f"Spatial action embedding requires num_actions ({num_actions}) == h * w ({h}*{w})"
        self.num_actions = num_actions
        self.h = h
        self.w = w
        self.embedding_dim = embedding_dim
        self.single_action_plane = single_action_plane
        self.projection = (
            nn.Conv2d(1, embedding_dim, kernel_size=1, bias=False)
            if single_action_plane
            else None
        )
        print(f"DEBUG_AGP_EMB: Using SpatialActionEmbedding ({h}x{w}) for {num_actions} actions.")

    def forward(self, action: Tensor) -> Tensor:
        """
        Args:
            action: One-hot encoded action tensor of shape [B, num_actions]
        Returns:
            Spatial embedding of shape [B, 1, H, W]
        """
        if not self.single_action_plane:
            # action is [B, num_actions] (one-hot)
            # We reshape to [B, 1, H, W]
            # Since action is one-hot, this puts a 1.0 at the correct (x, y) position
            return action.view(-1, 1, self.h, self.w)

        # NOTE: Old MuZero parity testing only. Legacy deterministic spatial
        # dynamics used a single normalized action plane projected with a 1x1 conv
        # instead of a localized one-hot board map.
        batch_size = action.shape[0]
        indices = torch.arange(
            self.num_actions, device=action.device, dtype=action.dtype
        )
        scalar_action = (action * indices).sum(dim=1) / self.num_actions
        action_plane = scalar_action.view(batch_size, 1, 1, 1).expand(
            batch_size, 1, self.h, self.w
        )
        return self.projection(action_plane)
