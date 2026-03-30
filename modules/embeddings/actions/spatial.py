import torch
from torch import nn
from torch import Tensor
from typing import Tuple


class SpatialActionEmbedding(nn.Module):
    """Encodes discrete actions as a spatial map on an (H, W) grid.

    For board games where ``num_actions == H * W``, the one-hot action vector
    is reshaped into a ``(B, 1, H, W)`` plane with a 1.0 at the action's
    board position.  A learnable ``Conv2d(1, embedding_dim, 1)`` then projects
    this sparse map to ``embedding_dim`` feature channels, giving the
    downstream fusion layer richer action information to work with.
    """
    def __init__(
        self,
        num_actions: int,
        h: int,
        w: int,
        embedding_dim: int = 1,
    ):
        super().__init__()
        assert (
        self.w = w
        self.embedding_dim = embedding_dim

        # Learnable projection from 1-channel spatial one-hot to embedding_dim channels.
        # When embedding_dim == 1 this is a trivial (but still learnable) scaling;
        # when embedding_dim > 1 it matches old MuZero's conv1x1(1 -> action_embedding_dim).
        self.projection = nn.Conv2d(1, embedding_dim, kernel_size=1, bias=False)

    def forward(self, action: Tensor) -> Tensor:
        """
        Args:
            action: One-hot encoded action tensor of shape ``[B, num_actions]``.

        Returns:
            Spatial embedding of shape ``[B, embedding_dim, H, W]``.
        """
        # Reshape one-hot → localized spatial map: (B, 1, H, W)
        spatial = action.view(-1, 1, self.h, self.w)
        return self.projection(spatial)
