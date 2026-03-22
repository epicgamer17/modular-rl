from typing import Tuple, Optional, Dict, Any
import torch
from torch import Tensor
import torch.nn.functional as F
from .base import BaseHead, HeadOutput
from agents.learner.losses.representations import BaseRepresentation, ScalarRepresentation, ClassificationRepresentation
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig
from modules.backbones.factory import BackboneFactory
from modules.backbones.mlp import build_dense, NoisyLinear


class ContinuationHead(BaseHead):
    """
    Predicts if the episode should continue (1.0) or end (0.0).
    Commonly used in Dreamer (gamma predictor) or general RL for termination prediction.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        representation: Optional[BaseRepresentation] = None,
        neck_config: Optional[BackboneConfig] = None,
        name: Optional[str] = None,
        input_source: str = "default",
    ):
        if representation is None:
            representation = ScalarRepresentation()
        super().__init__(arch_config, input_shape, representation, neck_config, name=name, input_source=input_source)

        # 1. Heads now build their own feature architecture (neck)
        self.neck = BackboneFactory.create(neck_config, input_shape)
        self.output_shape = self.neck.output_shape
        self.flat_dim = self._get_flat_dim(self.output_shape)

        # 2. Heads now define their own Final Output layer
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
        **kwargs,
    ) -> HeadOutput:
        """Returns HeadOutput with (logits, continuation_probability, state)"""
        # 1. Processing neck -> flatten
        x = self.neck(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)

        # 2. Final Output Projection
        logits = self.output_layer(x)

        # 3. Mathematical Transform
        continuation = self.representation.to_expected_value(logits)

        # Special casing for Dreamer-style binary continuation logic
        if (
            isinstance(self.representation, ClassificationRepresentation)
            and self.representation.num_features == 2
        ):
            probs = F.softmax(logits, dim=-1)
            continuation = probs[..., 1]  # Probability of "continue"

        return HeadOutput(
            training_tensor=logits,
            inference_tensor=continuation,
            state=state if state is not None else {},
        )
