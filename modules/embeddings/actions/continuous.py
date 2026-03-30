import torch
from torch import nn
from torch import Tensor

class ContinuousActionEmbedding(nn.Module):
    """
    Projects continuous action vectors to a fixed embedding dimension.
    """
    def __init__(self, action_dim: int, embedding_dim: int):
        super().__init__()
        self.num_actions = action_dim
        self.projection = nn.Linear(action_dim, embedding_dim)
        self.activation = nn.ReLU()
        print(f"DEBUG_AGP_EMB: Using ContinuousActionEmbedding for {action_dim} dim actions (emb={embedding_dim}).")

    def forward(self, action: Tensor) -> Tensor:
        """
        Args:
            action: Continuous action tensor of shape [B, action_dim]
        Returns:
            Embedding of shape [B, embedding_dim]
        """
        return self.activation(self.projection(action))
