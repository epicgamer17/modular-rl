from typing import Tuple, Optional, Dict, Any
from torch import nn, Tensor
from .base import BaseHead
from agents.learner.losses.representations import (
    BaseRepresentation,
    IdentityRepresentation,
)
from modules.blocks.linear import build_linear_block


class LatentConsistencyHead(BaseHead):
    """
    Projects latent states into a embedding space for consistency loss.
    Commonly used in MuZero (projection head) or Dreamer.
    Typically a small MLP.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        representation: Optional[BaseRepresentation] = None,
        neck: Optional[nn.Module] = None,
        noisy_sigma: float = 0.0,
        projection_dim: int = 256,
    ):
        # If representation is None, default to IdentityRepresentation with projection_dim
        if representation is None:
            representation = IdentityRepresentation(num_features=projection_dim)

        super().__init__(input_shape, representation, neck, noisy_sigma)
        self.projection_dim = projection_dim

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any], Tensor]:
        state = state if state is not None else {}

        # Apply neck if it exists
        if self.neck is not None:
            x = self.neck(x)

        # Apply the output layer to get logits
        logits = self.output_layer(x)

        # The projected latent
        projected_embedding = self.representation.to_expected_value(logits)

        return logits, state, projected_embedding
