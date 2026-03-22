from typing import Tuple, Optional, Dict, Any
from torch import Tensor
import torch.nn.functional as F
from .base import BaseHead, HeadOutput
from agents.learner.losses.representations import BaseRepresentation, ScalarRepresentation, ClassificationRepresentation
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig


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
    ):
        # Default to ScalarRepresentation(1) if none provided, but often used as ClassificationRepresentation(2)
        if representation is None:
            representation = ScalarRepresentation()

        super().__init__(arch_config, input_shape, representation, neck_config)

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> HeadOutput:
        """Returns HeadOutput with (logits, continuation_probability, state)"""
        state = state if state is not None else {}
        head_out = super().forward(x, state)
        logits = head_out.training_tensor

        # continuation is essentially the expected value (probability if classification)
        continuation = self.representation.to_expected_value(logits)

        # If it's classification(2), we want the probability of class 1
        if (
            isinstance(self.representation, ClassificationRepresentation)
            and self.representation.num_features == 2
        ):
            probs = F.softmax(logits, dim=-1)
            continuation = probs[..., 1]  # Probability of "continue"

        return HeadOutput(
            training_tensor=logits,
            inference_tensor=continuation,
            state=state,
        )
