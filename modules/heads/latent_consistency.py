from typing import Tuple, Optional, Dict, Any
from torch import Tensor
from .base import BaseHead, HeadOutput
from agents.learner.losses.representations import (
    BaseRepresentation,
    IdentityRepresentation,
)
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig
from modules.backbones.mlp import build_dense


class LatentConsistencyHead(BaseHead):
    """
    Projects latent states into a embedding space for consistency loss.
    Commonly used in MuZero (projection head) or Dreamer.
    Typically a small MLP.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        representation: Optional[BaseRepresentation] = None,
        neck_config: Optional[BackboneConfig] = None,
        projection_dim: int = 256,
    ):
        # If representation is None, default to IdentityRepresentation with projection_dim
        if representation is None:
            representation = IdentityRepresentation(num_features=projection_dim)

        super().__init__(arch_config, input_shape, representation, neck_config)
        self.projection_dim = projection_dim

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> HeadOutput:
        state = state if state is not None else {}

        # Apply neck if it exists
        if self.neck is not None:
            x = self.neck(x)

        # Apply the output layer to get logits
        logits = self.output_layer(x)

        # The projected latent
        projected_embedding = self.representation.to_expected_value(logits)

        return HeadOutput(
            training_tensor=logits,
            inference_tensor=projected_embedding,
            state=state,
        )
