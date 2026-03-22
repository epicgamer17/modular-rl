from typing import Tuple, Optional, Dict, Any
import torch
from torch import Tensor
from .base import BaseHead, HeadOutput
from agents.learner.losses.representations import (
    ClassificationRepresentation,
)
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig
from modules.backbones.factory import BackboneFactory
from modules.backbones.mlp import build_dense, NoisyLinear


class ChanceProbabilityHead(BaseHead):
    """
    Predicts the probability distribution over chance outcomes (codes).
    Used in Stochastic MuZero.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        num_chance_codes: int,
        neck_config: Optional[BackboneConfig] = None,
        name: Optional[str] = None,
        input_source: str = "default",
    ):
        representation = ClassificationRepresentation(num_classes=num_chance_codes)
        super().__init__(arch_config, input_shape, representation, neck_config, name=name, input_source=input_source)

        # 1. Heads now build their own feature architecture (neck)
        self.neck = BackboneFactory.create(neck_config, input_shape)
        self.output_shape = self.neck.output_shape
        self.flat_dim = self._get_flat_dim(self.output_shape)

        # 2. Heads now define their own Final Output layer
        # Typically a categorization over N codes.
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
        """Returns HeadOutput with (chance_logits, distribution, state)"""
        # 1. Processing neck -> flatten
        x = self.neck(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)

        # 2. Final Output Projection
        logits = self.output_layer(x)

        # 3. Mathematical Transform (Categorical distribution)
        inference = None
        if is_inference:
            inference = self.representation.to_inference(logits)

        return HeadOutput(
            training_tensor=logits,
            inference_tensor=inference,
            state=state if state is not None else {},
        )
