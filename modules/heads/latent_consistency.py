from typing import Tuple, Optional, Dict, Any, Callable
import torch
from torch import nn, Tensor
from .base import BaseHead, HeadOutput
from configs.modules.heads.latent_consistency import SimSiamProjectorConfig
from agents.learner.losses.representations import (
    BaseRepresentation,
    IdentityRepresentation,
)
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig
from modules.backbones.mlp import build_dense, NoisyLinear




class SimSiamProjectorHead(BaseHead):
    """
    Consolidated SimSiam/BYOL Projection head.
    Matches the architecture from modules/projectors/sim_siam.py.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        proj_hidden_dim: int,
        proj_output_dim: int,
        pred_hidden_dim: int,
        pred_output_dim: int,
        representation: Optional[BaseRepresentation] = None,
        neck_fn: Optional[Callable[[Tuple[int, ...]], nn.Module]] = None,
        noisy_sigma: float = 0.0,
        name: Optional[str] = None,
        input_source: str = "default",
        **kwargs,
    ):
        if representation is None:
            representation = IdentityRepresentation(num_features=pred_output_dim)

        super().__init__(
            input_shape,
            representation,
            neck_fn=neck_fn,
            noisy_sigma=noisy_sigma,
            name=name,
            input_source=input_source,
        )

        # 1. Heads now build their own feature architecture (neck)
        if self.neck_fn is not None:
            self.neck = self.neck_fn(input_shape=input_shape)
        else:
            self.neck = nn.Identity()
            self.neck.output_shape = input_shape

        self.output_shape = self.neck.output_shape
        self.flat_dim = self._get_flat_dim(self.neck, input_shape)

        # 2. Projection layers (SimSiam/EfficientZero style)
        self.projection = nn.Sequential(
            nn.Linear(self.flat_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.BatchNorm1d(proj_output_dim),
        )

        # 3. Predictor layers (The 'projection_head' in original code)
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, pred_output_dim),
        )

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
        is_inference: bool = False,
        **kwargs,
    ) -> HeadOutput:
        """Returns HeadOutput with (prediction, projection, state)"""
        # 1. Neck + Flatten
        x = self.neck(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)

        # 2. SimSiam Forward pass
        # Note: BatchNorm1d requires batch_size > 1.
        # This is strictly a training-only head in most MuZero variants.
        projected = self.projection(x)
        predicted = self.predictor(projected)

        return HeadOutput(
            training_tensor=predicted,
            inference_tensor=projected,  # We store projection in inference_tensor
            state=state if state is not None else {},
        )
