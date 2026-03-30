import torch
from torch import nn
from torch import Tensor
from typing import Tuple

class SpatialActionEmbedding(nn.Module):
    """
    Encodes actions as a 1.0 at the (x, y) coordinate on a spatial grid.
    Used for board games like Tic-Tac-Toe, Go, Chess where actions map to board positions.
    """
    def __init__(self, num_actions: int, h: int, w: int):
        super().__init__()
        assert num_actions == h * w, f"Spatial action embedding requires num_actions ({num_actions}) == h * w ({h}*{w})"
        self.num_actions = num_actions
        self.h = h
        self.w = w
        print(f"DEBUG_AGP_EMB: Using SpatialActionEmbedding ({h}x{w}) for {num_actions} actions.")

    def forward(self, action: Tensor) -> Tensor:
        """
        Args:
            action: One-hot encoded action tensor of shape [B, num_actions]
        Returns:
            Spatial embedding of shape [B, 1, H, W]
        """
        # action is [B, num_actions] (one-hot)
        # We reshape to [B, 1, H, W]
        # Since action is one-hot, this puts a 1.0 at the correct (x, y) position
        return action.view(-1, 1, self.h, self.w)
