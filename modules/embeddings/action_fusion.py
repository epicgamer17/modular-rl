import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

from modules.embeddings.action_embedding import ActionEncoder

class ActionFusion(nn.Module):
    """Embeds an action and fuses it into a latent state."""
    def __init__(self, num_actions: int, embedding_dim: int, input_shape: Tuple[int, ...]):
        super().__init__()
        self.num_actions = num_actions
        self.encoder = ActionEncoder(num_actions, embedding_dim)
        
        # If image (C, H, W), use Conv. If vector (D,), use Linear.
        if len(input_shape) == 3:
            # Spatial: input_shape[0] is channels
            self.fusion = nn.Conv2d(input_shape[0] + embedding_dim, input_shape[0], 3, padding=1)
        else:
            # Vector: input_shape[0] is dimension
            self.fusion = nn.Linear(input_shape[0] + embedding_dim, input_shape[0])

    def forward(self, latent: Tensor, action: Tensor) -> Tensor:
        # 1. Handle discrete action indexing and one-hot conversion
        if action.dtype in [torch.long, torch.int]:
            if action.dim() > 1 and action.shape[-1] == 1:
                action = action.squeeze(-1)
            if action.dim() == 1:
                action = F.one_hot(action, num_classes=self.num_actions).float()
        
        # 2. Match device of latent state
        action = action.to(latent.device)

        # 3. Embed and Expand (ActionEncoder handles matching spatial dimensions)
        action_emb = self.encoder(action, latent.shape)
        
        # 4. Fuse using the allocated layer (Conv2d or Linear)
        x = torch.cat([latent, action_emb], dim=1)
        
        # 5. Residual connection + ReLU (Protects gradient flow)
        return F.relu(self.fusion(x) + latent)
