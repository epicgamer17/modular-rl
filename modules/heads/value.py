from typing import Tuple, Optional, Dict, Any
from torch import Tensor
from .base import BaseHead, HeadOutput
from agents.learner.losses.representations import BaseRepresentation
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig


class ValueHead(BaseHead):
    """
    Predicts the expected return (Value).
    Supports multiple output strategies (Scalar, MuZero, C51, Dreamer).
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
        """Returns HeadOutput with (logits, expected_value, state)"""
        head_out = super().forward(x, state)
        expected_value = self.representation.to_expected_value(head_out.training_tensor)
        return HeadOutput(
            training_tensor=head_out.training_tensor,
            inference_tensor=expected_value,
            state=head_out.state,
        )
