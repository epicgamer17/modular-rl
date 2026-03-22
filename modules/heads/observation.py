from typing import Tuple, Optional, Dict, Any
import torch
from torch import Tensor
from .base import BaseHead, HeadOutput
from agents.learner.losses.representations import BaseRepresentation, ScalarRepresentation
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig
from modules.backbones.factory import BackboneFactory
from modules.backbones.mlp import build_dense, NoisyLinear


class ObservationHead(BaseHead):
    """
    Predicts/reconstructs the observation.
    Can be used for Dreamer decoders or general autoencoding/prediction tasks.
    Flexible enough to handle image reconstruction (via neck=DeconvBackbone)
    or vector reconstruction (via neck=MLPBackbone).
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        representation: Optional[BaseRepresentation] = None,
        neck_config: Optional[BackboneConfig] = None,
        use_output_layer: bool = True,
    ):
        super().__init__(arch_config, input_shape, representation, neck_config)

        # 1. Image or Vector neck (e.g. Deconv or MLP)
        self.neck = BackboneFactory.create(neck_config, input_shape)
        self.output_shape = self.neck.output_shape
        self.flat_dim = self._get_flat_dim(self.output_shape)

        self.use_output_layer = use_output_layer
        self.output_layer = None

        # 2. Representation setup
        if self.representation is None:
            self.representation = representation or ScalarRepresentation()

        # 3. Optional Final Dense Output Layer
        if self.use_output_layer:
            self.output_layer = build_dense(
                in_features=self.flat_dim,
                out_features=self.representation.num_features,
                sigma=self.arch_config.noisy_sigma,
            )

    def reset_noise(self) -> None:
        """Propagate noise reset through the neck and optional output layer."""
        if hasattr(self.neck, "reset_noise"):
            self.neck.reset_noise()
        if isinstance(self.output_layer, NoisyLinear):
            self.output_layer.reset_noise()

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> HeadOutput:
        """Returns HeadOutput with (logits/features, observation_pred, state)"""
        # 1. Process neck (Deconv for images, MLP for vectors)
        x = self.neck(x)

        # 2. Optional Flat Dense Projection
        if self.output_layer is not None:
            if x.dim() > 2:
                x = x.flatten(1, -1)
            logits = self.output_layer(x)
        else:
            # If no output layer, the neck output serves as the prediction/features
            logits = x

        # 3. Mathematical Transform (e.g. Pixel normalization or HL-Gauss)
        observation_pred = self.representation.to_expected_value(logits)

        return HeadOutput(
            training_tensor=logits,
            inference_tensor=observation_pred,
            state=state if state is not None else {},
        )
