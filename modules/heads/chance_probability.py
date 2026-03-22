from typing import Tuple, Optional, Dict, Any
from torch import Tensor
from .base import BaseHead, HeadOutput
from agents.learner.losses.representations import ClassificationRepresentation
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig


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
    ):
        representation = ClassificationRepresentation(num_classes=num_chance_codes)
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
