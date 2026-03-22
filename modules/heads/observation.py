from typing import Tuple, Optional, Dict, Any
from torch import Tensor
import torch
from .base import BaseHead, HeadOutput
from agents.learner.losses.representations import BaseRepresentation, ScalarRepresentation
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig


class ObservationHead(BaseHead):
    """
    Predicts/reconstructs the observation.
    Can be used for Dreamer decoders or general autoencoding/prediction tasks.
    Flexible enough to handle image reconstruction (via neck=DeconvBackbone)
    or vector reconstruction (via neck=DenseBackbone).
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        representation: Optional[BaseRepresentation] = None,
        neck_config: Optional[BackboneConfig] = None,
        use_output_layer: bool = True,
    ):
        # Pass representation=None to avoid creating the default output layer in BaseHead if not wanted
        super().__init__(
            arch_config,
            input_shape,
            representation if use_output_layer else None,
            neck_config,
        )
        self.use_output_layer = use_output_layer

        # Ensure we have a representation for logic downstream
        if self.representation is None:
            self.representation = representation or ScalarRepresentation()

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> HeadOutput:
        """Returns HeadOutput with (logits, observation_pred, state)"""
        state = state if state is not None else {}

        # Process neck
        x = self.neck(x)

        # If we have a final output layer (dense), we project.
        if self.output_layer is not None:
            if x.dim() > 2:
                x = x.flatten(1, -1)
            logits = self.output_layer(x)
        else:
            logits = x

        # expected_observation is representation conversion
        observation_pred = self.representation.to_expected_value(logits)

        return HeadOutput(
            training_tensor=logits,
            inference_tensor=observation_pred,
            state=state,
        )
