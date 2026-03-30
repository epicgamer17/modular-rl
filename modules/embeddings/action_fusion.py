import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from modules.embeddings.action_embedding import ActionEncoder


class ActionFusion(nn.Module):
    """Embeds an action and fuses it into a latent state."""

    def __init__(
        self, encoder: ActionEncoder, input_shape: Tuple[int, ...], use_bn: bool = False
    ):
        super().__init__()
        self.encoder = encoder
        embedding_dim = encoder.embedding_dim

        # If image (C, H, W), use Conv. If vector (D,), use Linear.
        if len(input_shape) == 3:
            # Spatial: input_shape[0] is channels
            self.fusion = nn.Conv2d(
                input_shape[0] + embedding_dim,
                input_shape[0],
                3,
                padding=1,
                bias=not use_bn,
            )
            self.bn = nn.BatchNorm2d(input_shape[0]) if use_bn else nn.Identity()
        else:
            # Vector: input_shape[0] is dimension
            self.fusion = nn.Linear(
                input_shape[0] + embedding_dim, input_shape[0], bias=not use_bn
            )
            self.bn = nn.BatchNorm1d(input_shape[0]) if use_bn else nn.Identity()

    def forward(self, latent: Tensor, action: Tensor) -> Tensor:
        # 1. Handle discrete action indexing and one-hot conversion
        # Use action_space_size if available in the encoder's embedding module
        # But for now, we assume the action is already one-hot if discrete or raw if continuous.
        # Actually, let's keep the helper logic but try to be more robust.
        if action.dtype in [torch.long, torch.int]:
            if action.dim() > 1 and action.shape[-1] == 1:
                action = action.squeeze(-1)
            # We need num_classes here. We'll try to find it in the encoder's module.
            num_classes = self.encoder.embedding_module.num_actions
            if num_classes is None:
                # Fallback to a reasonable heuristic if not found
                # Or we can just assume it's already one-hot if it's long? No, usually indices are long.
                raise ValueError(
                    "Action is discrete but encoder has no 'num_actions' attribute. Cannot one-hot."
                )

            if action.dim() == 1:
                action = F.one_hot(action, num_classes=num_classes).float()

        # 2. Match device of latent state
        action = action.to(latent.device)

        # 3. Embed and Expand (ActionEncoder handles matching spatial dimensions)
        action_emb = self.encoder(action, latent.shape)

        # 4. Fuse using the allocated layer (Conv2d or Linear)
        x = torch.cat([latent, action_emb], dim=1)
        x = self.fusion(x)
        x = self.bn(x)

        # 5. Residual connection + ReLU (Protects gradient flow)
        return F.relu(x + latent)
