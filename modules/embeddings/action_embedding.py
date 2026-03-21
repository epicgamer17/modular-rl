import torch
from torch import nn
from typing import Tuple, Union, Optional


class ActionEncoder(nn.Module):
    """
    Standardizes action encoding across vector and spatial domains.
    Recursively projects actions into a latent embedding that matches either 
    a flat vector state or a spatial feature map.

    Args:
        action_space_size: Number of discrete actions or dimension of continuous actions.
        embedding_dim: Output channel/feature dimension of the embedding.
    """

    def __init__(
        self,
        action_space_size: int,
        embedding_dim: int = 32,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.embedding_dim = embedding_dim

        # Standard linear projection for both discrete (one-hot) and continuous actions.
        # This replaces algorithms-specific branching and soft-indexing logic.
        self.encoder = nn.Linear(action_space_size, embedding_dim, bias=False)

    def forward(
        self, action: torch.Tensor, target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Encodes actions into the target domain shape.

        Args:
            action: (B, A) tensor where A is action_space_size. 
                    Expects one-hot encoded discrete actions or raw continuous actions.
            target_shape: The shape of the destination hidden state to match.
                          Length 2 -> Vector output (B, D)
                          Length 4 -> Spatial output (B, D, H, W)

        Returns:
            Encoded action tensor expanded to match the spatial or flat dimensions of the target.
        """
        # 1. Project to embedding dimension
        # [B, A] -> [B, D]
        x = self.encoder(action)

        # 2. Match target dimensionality
        ndim = len(target_shape)

        if ndim == 2:
            # Flat vector: already correct shape (B, D)
            return x

        elif ndim == 4:
            # Spatial map: Expand [B, D] -> [B, D, 1, 1] -> [B, D, H, W]
            h, w = target_shape[2], target_shape[3]
            return x.view(-1, self.embedding_dim, 1, 1).expand(-1, -1, h, w)

        else:
            raise ValueError(
                f"ActionEncoder target_shape must be length 2 or 4, got {target_shape}"
            )
