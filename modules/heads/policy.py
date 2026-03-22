from typing import Tuple, Optional, Dict, Any
import torch
from torch import Tensor
from .base import BaseHead, HeadOutput
from agents.learner.losses.representations import BaseRepresentation
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig


class PolicyHead(BaseHead):
    """
    Predicts the action distribution (Policy).
    Supports both discrete (Categorical) and continuous (Gaussian) actions via Representation.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        representation: BaseRepresentation,
        neck_config: Optional[BackboneConfig] = None,
    ):
        super().__init__(arch_config, input_shape, representation, neck_config)

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> HeadOutput:
        """Returns HeadOutput with (logits, dist_obj, state)"""
        head_out = super().forward(x, state)
        inference = self.representation.to_inference(head_out.training_tensor)
        return HeadOutput(
            training_tensor=head_out.training_tensor,
            inference_tensor=inference,
            state=head_out.state,
        )
