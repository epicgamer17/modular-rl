from typing import Tuple, Optional, Dict, Any, Callable
import torch
from torch import Tensor
import torch.nn as nn
from .base import BaseHead, HeadOutput
from agents.learner.losses.representations import (
    ClassificationRepresentation,
)
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig
from modules.backbones.mlp import build_dense, NoisyLinear


class ChanceProbabilityHead(BaseHead):
    """
    Predicts the probability distribution over chance outcomes (codes).
    Used in Stochastic MuZero.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_chance_codes: int,
        neck_fn: Optional[Callable[[Tuple[int, ...]], nn.Module]] = None,
        noisy_sigma: float = 0.0,
        name: Optional[str] = None,
        input_source: str = "default",
        **kwargs,
    ):
        representation = ClassificationRepresentation(num_classes=num_chance_codes)
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

        # 2. Heads now define their own Final Output layer
        # Typically a categorization over N codes.
        self.output_layer = build_dense(
            in_features=self.flat_dim,
            out_features=self.representation.num_features,
            sigma=self.noisy_sigma,
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
