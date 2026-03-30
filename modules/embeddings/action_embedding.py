import torch
from torch import nn
from typing import Tuple, Union, Optional


class ActionEncoder(nn.Module):
    """
    Standardizes action encoding across vector and spatial domains.
    Delegates core embedding logic to a specialized module (Spatial, Continuous, or EfficientZero).

    Args:
        embedding_module: The neural network module that performs the initial embedding.
        embedding_dim: The number of channels or features produced by the embedding_module.
    """

    def __init__(
        self,
        embedding_module: nn.Module,
        embedding_dim: int,
    ):
        super().__init__()
        self.embedding_module = embedding_module
        self.embedding_dim = embedding_dim

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
        # 1. Delegate core embedding
        # Result could be [B, D] or [B, D, H, W]
        x = self.embedding_module(action)

        # 2. Match target dimensionality
        ndim_target = len(target_shape)
        ndim_output = x.dim()

        if ndim_target == 2:
            # Target is vector (B, D)
            return x

        elif ndim_target == 4:
            # Target is spatial (B, D, H, W)
            if ndim_output == 4:
                # Already spatial, assume it matches H, W of target
                return x
            else:
                # Output is vector (B, D), expand to [B, D, H, W]
                h, w = target_shape[2], target_shape[3]
                return x.view(-1, self.embedding_dim, 1, 1).expand(-1, -1, h, w)

        else:
            raise ValueError(
                f"ActionEncoder target_shape must be length 2 or 4, got {target_shape}"
            )


def get_action_encoder(
    num_actions: int,
    latent_dimensions: Tuple[int, ...],
    is_discrete: bool = True,
    action_embedding_dim: int = 16,
    is_spatial: Optional[bool] = None,
) -> ActionEncoder:
    """
    Factory to select and construct the correct ActionEncoder.

    Args:
        num_actions: Size of action space (or action_dim if continuous).
        latent_dimensions: Shape of the hidden state (C, H, W) or (D,).
        is_discrete: Whether actions are discrete or continuous.
        action_embedding_dim: Output dimension for vector-based embeddings.
        is_spatial: If True, forces SpatialActionEmbedding. If False, forces non-spatial.
                   If None, infers from latent_dimensions and num_actions.
    """
    from modules.embeddings.actions.spatial import SpatialActionEmbedding
    from modules.embeddings.actions.continuous import ContinuousActionEmbedding
    from modules.embeddings.actions.efficient_zero import EfficientZeroActionEmbedding

    # 1. Selection Logic
    use_spatial = False
    if is_spatial is True:
        use_spatial = True
    elif is_spatial is False:
        use_spatial = False
    else:
        # Heuristic: If discrete and latent is spatial and num_actions == spatial_grid_size
        if is_discrete and len(latent_dimensions) == 3:
            h, w = latent_dimensions[1], latent_dimensions[2]
            if num_actions == h * w:
                use_spatial = True

    # 2. Construction
    if use_spatial:
        h, w = latent_dimensions[1], latent_dimensions[2]
        module = SpatialActionEmbedding(num_actions, h, w)
        # Spatial embedding always produces 1 channel, so dim=1
        return ActionEncoder(module, embedding_dim=1)
    
    if not is_discrete:
        module = ContinuousActionEmbedding(num_actions, action_embedding_dim)
        return ActionEncoder(module, embedding_dim=action_embedding_dim)
    
    # Default: EfficientZero style discrete projection
    module = EfficientZeroActionEmbedding(num_actions, action_embedding_dim)
    return ActionEncoder(module, embedding_dim=action_embedding_dim)
