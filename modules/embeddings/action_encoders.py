import torch
from torch import nn
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class BaseActionEncoder(nn.Module, ABC):
    """Base interface for action encoders."""

    def __init__(self, action_space_size: int, embedding_dim: int):
        super().__init__()
        self.action_space_size = action_space_size
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, action: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        pass


class VectorActionEncoder(BaseActionEncoder):
    """Encodes actions into a vector embedding."""

    def __init__(self, action_space_size: int, embedding_dim: int, bias: bool = False):
        super().__init__(action_space_size, embedding_dim)
        self.fc = nn.Linear(action_space_size, embedding_dim, bias=bias)

    def forward(self, action: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        # action: (B, A) [one-hot or soft]
        return self.fc(action)


class ImageActionEncoder(BaseActionEncoder):
    """Encodes actions into a spatial embedding (planes)."""

    def __init__(
        self,
        action_space_size: int,
        embedding_dim: int,
        single_action_plane: bool = True,
        bias: bool = False,
    ):
        super().__init__(action_space_size, embedding_dim)
        self.single_action_plane = single_action_plane
        
        in_channels = 1 if single_action_plane else action_space_size
        self.conv = nn.Conv2d(in_channels, embedding_dim, kernel_size=1, bias=bias)

    def forward(self, action: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        # target_shape: (B, C, H, W)
        batch_size, _, h, w = target_shape
        device = action.device
        
        if self.single_action_plane:
            # Create a plane scaled by action index (soft discrete logic)
            # Expecting action (B, A)
            indices = torch.arange(self.action_space_size, device=device, dtype=torch.float32)
            scalar_action = torch.sum(action * indices, dim=1) / self.action_space_size
            action_place = scalar_action.view(batch_size, 1, 1, 1).expand(batch_size, 1, h, w)
        else:
            # Expand (B, A) -> (B, A, H, W)
            action_place = action.view(batch_size, self.action_space_size, 1, 1).expand(
                batch_size, self.action_space_size, h, w
            )
            
        return self.conv(action_place)


class ActionEncoder(BaseActionEncoder):
    """
    Unified action encoder that dispatches to Image or Vector implementation.
    """

    def __init__(
        self,
        action_space_size: int,
        embedding_dim: int = 32,
        is_continuous: bool = False,
        single_action_plane: bool = True,
    ):
        super().__init__(action_space_size, embedding_dim)
        self.is_continuous = is_continuous
        
        self.vector_encoder = VectorActionEncoder(action_space_size, embedding_dim)
        self.image_encoder = ImageActionEncoder(
            action_space_size, embedding_dim, single_action_plane=single_action_plane
        )

    def forward(self, action: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        ndim = len(target_shape)
        if ndim == 4:
            return self.image_encoder(action, target_shape)
        elif ndim == 2:
            return self.vector_encoder(action, target_shape)
        else:
            raise ValueError(f"Unsupported target shape: {target_shape}")
