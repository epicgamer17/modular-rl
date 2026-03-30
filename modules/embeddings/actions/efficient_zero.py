import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

class EfficientZeroActionEmbedding(nn.Module):
    """
    Encodes discrete actions using 'soft indices' or a standard projection.
    Based on the EfficientZero/MuZero approach for discrete action spaces.
    """
    def __init__(self, num_actions: int, embedding_dim: int):
        super().__init__()
        self.num_actions = num_actions
        self.projection = nn.Linear(num_actions, embedding_dim)
        # Often includes normalization or extra layers in production MuZero
        self.norm = nn.LayerNorm(embedding_dim)
        print(f"DEBUG_AGP_EMB: Using EfficientZeroActionEmbedding for {num_actions} actions (emb={embedding_dim}).")

    def forward(self, action: Tensor) -> Tensor:
        """
        Args:
            action: One-hot encoded action tensor of shape [B, num_actions]
        Returns:
            Embedding of shape [B, embedding_dim]
        """
        x = self.projection(action)
        return self.norm(x)
