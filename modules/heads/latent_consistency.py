from typing import Tuple, Optional, Dict, Any
import torch
from torch import Tensor
from .base import BaseHead, HeadOutput
from agents.learner.losses.representations import (
    BaseRepresentation,
    IdentityRepresentation,
)
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig
from modules.backbones.factory import BackboneFactory
from modules.backbones.mlp import build_dense, NoisyLinear


class LatentConsistencyHead(BaseHead):
    """
    Projects latent states into an embedding space for consistency loss.
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
        name: Optional[str] = None,
        input_source: str = "default",
    ):
        if representation is None:
            representation = IdentityRepresentation(num_features=projection_dim)

        super().__init__(arch_config, input_shape, representation, neck_config, name=name, input_source=input_source)
        self.projection_dim = projection_dim

        # 1. Heads now build their own feature architecture (neck)
        self.neck = BackboneFactory.create(neck_config, input_shape)
        self.output_shape = self.neck.output_shape
        self.flat_dim = self._get_flat_dim(self.output_shape)

        # 2. Heads now define their own Final Output layer
        # For Consistency, this is usually a projection to projection_dim
        self.output_layer = build_dense(
            in_features=self.flat_dim,
            out_features=self.representation.num_features,
            sigma=self.arch_config.noisy_sigma,
        )

    def reset_noise(self) -> None:
        """Propagate noise reset through the head's submodules."""
        if hasattr(self.neck, "reset_noise"):
            self.neck.reset_noise()
        if isinstance(self.output_layer, NoisyLinear):
            self.output_layer.reset_noise()

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
        is_inference: bool = False,
        **kwargs,
    ) -> HeadOutput:
        """Returns HeadOutput with (projected_logits, consistency_embedding, state)"""
        # 1. Processing neck -> flatten
        x = self.neck(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)

        # 2. Final Output Projection
        logits = self.output_layer(x)

        # 3. Mathematical Transform (Identity for consistency usually)
        projected_embedding = None
        if is_inference:
            projected_embedding = self.representation.to_expected_value(logits)

        return HeadOutput(
            training_tensor=logits,
            inference_tensor=projected_embedding,
            state=state if state is not None else {},
        )
