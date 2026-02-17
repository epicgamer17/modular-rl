import torch
from torch import nn
from typing import Tuple


class ActionEncoder(nn.Module):
    def __init__(
        self,
        action_space_size: int,
        embedding_dim: int = 32,
        is_continuous: bool = False,
        single_action_plane: bool = True,
    ):
        """
        Encodes actions into an embedding compatible with the hidden state.

        Args:
            action_space_size: Number of discrete actions or dimension of continuous action.
            embedding_dim: Output channel/feature dimension of the embedding.
            is_continuous: Whether the action space is continuous (float) or discrete (int).
        """
        super().__init__()
        self.action_space_size = action_space_size
        self.embedding_dim = embedding_dim
        self.is_continuous = is_continuous
        self.single_action_plane = single_action_plane
        assert not (single_action_plane and is_continuous)

        # --- Image Path Layers (4D) ---
        # Based on reference: Discrete actions are 1 plane (scaled), Continuous are N planes.
        in_channels_image = (
            action_space_size if is_continuous or (not single_action_plane) else 1
        )

        self.conv1x1 = nn.Conv2d(
            in_channels_image, embedding_dim, kernel_size=1, bias=False
        )
        # Using BatchNorm2d instead of LayerNorm allows variable H, W input sizes
        self.bn_image = nn.BatchNorm2d(embedding_dim)

        # --- Vector Path Layers (2D) ---
        self.fc_vector = nn.Linear(action_space_size, embedding_dim, bias=False)
        self.bn_vector = nn.BatchNorm1d(embedding_dim)

    def forward(self, action: torch.Tensor, target_shape: Tuple[int]):
        """
        Args:
            action: (B,) or (B, 1) IntTensor for discrete, (B, A) FloatTensor for continuous.
            target_shape: The shape of the hidden state to match.
                          Length 4 implies (B, C, H, W).
                          Length 2 implies (B, D).
        Returns:
            Encoded action tensor matching the spatial/flat dimensions of target_shape.
            Image: (B, embedding_dim, H, W)
            Vector: (B, embedding_dim)
        """
        ndim = len(target_shape)

        if ndim == 4:
            return self._forward_image(action, target_shape)
        elif ndim == 2:
            return self._forward_vector(action)
        else:
            raise ValueError(
                f"Target shape must be len 2 (vector) or 4 (image), got {target_shape}"
            )

    def _forward_image(self, action: torch.Tensor, shape: Tuple[int]):
        # shape is (B, C, H, W)
        batch_size, _, h, w = shape
        device = action.device

        # 1. Construct the Action Planes (Reference Logic)
        if not self.is_continuous and self.single_action_plane:
            # Discrete: Create a plane of ones, scale by action index / space_size
            # Expecting action (B,) or (B, 1)
            # if action.dim() == 1:
            #     action = action.view(-1, 1)  # Ensure (B, 1)

            # 2. Create "Soft Indices"
            # We create a vector [0, 1, 2, ..., N]
            if self.action_space_size != h * w:
                indices = torch.arange(
                    self.action_space_size, device=device, dtype=torch.float32
                )

                # Multiply (B, A) * (A,) -> (B, A) then Sum -> (B,)
                # If input is [0, 1, 0], we get scalar 1.0
                # If input is [0.1, 0.8, 0.1] (soft), we get scalar 1.0 (weighted avg)
                assert (
                    action.dim() == 2
                ), f"Action must be (B, A) for soft discrete encoding got {action.shape}"
                scalar_action = torch.sum(action * indices, dim=1)
                assert torch.allclose(
                    scalar_action, torch.argmax(action, dim=1).float()
                ), f"{scalar_action} vs {torch.argmax(action, dim=1).float()}"

                # 3. Normalize (Optional but recommended)
                # Keeps values between 0.0 and 1.0
                scalar_action = scalar_action / self.action_space_size

                # 4. Expand to 1 Plane
                # (B,) -> (B, 1, 1, 1) -> (B, 1, H, W)
                action_place = scalar_action.view(batch_size, 1, 1, 1).expand(
                    batch_size, 1, h, w
                )  # Broadcast multiply: (B, 1, 1, 1) * (B, 1, H, W)

            else:
                # print("using spatial")
                # TODO: REMOVE AFTER TESTING TIC TAC TOE:
                action_place = torch.zeros((batch_size, 1, h, w), device=device)

                action_indices = action.argmax(dim=1)  # Handle OneHot
                x_coords = action_indices % w
                y_coords = action_indices // w

                # Safety clip for robustness (though your logic should prevent this)
                y_coords = y_coords.clamp(0, h - 1)
                x_coords = x_coords.clamp(0, w - 1)

                action_place[
                    torch.arange(batch_size), 0, y_coords.long(), x_coords.long()
                ] = 1.0
        else:
            # print("using categorical")
            # Continuous: Expand (B, A) -> (B, A, H, W)
            # action shape (B, A)
            action_place = action.view(batch_size, self.action_space_size, 1, 1)
            action_place = action_place.expand(batch_size, self.action_space_size, h, w)

        # 2. Embed and Normalize
        x = self.conv1x1(action_place)
        # x = self.bn_image(x)
        # x = F.relu(x)

        return x

    def _forward_vector(self, action: torch.Tensor):
        # 1. Embed
        x = self.fc_vector(action)

        # 2. Normalize
        # x = self.bn_vector(x)
        # x = F.relu(x)

        return x
