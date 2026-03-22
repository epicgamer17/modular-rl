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
    Integrates explicit action masking into the compute graph for stable learning.
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
        action_mask: Optional[Tensor] = None,
    ) -> HeadOutput:
        """Returns HeadOutput containing masked logits and/or distribution object."""
        # 1. Standard processing (neck -> flatten)
        x = self.process_input(x)

        # 2. Output Projection Layer
        logits = self.output_layer(x)

        # 3. Apply Masking Logic (Logit-level)
        # We apply masking BEFORE packing into HeadOutput so the Learner's
        # standard CrossEntropy Loss automatically respects the mask.
        if action_mask is not None:
            # Valid actions remain, invalid shift to -1e8 (prob ~0)
            HUGE_NEG = torch.tensor(-1e8, dtype=logits.dtype, device=logits.device)
            logits = torch.where(action_mask.bool(), logits, HUGE_NEG)

            # Build specialized distribution for the Actor
            from modules.distributions import MaskedCategorical

            inference = MaskedCategorical(logits=logits, mask=action_mask)
        else:
            # Standard path (Gaussian or Categorical via Representation)
            inference = self.representation.to_inference(logits)

        return HeadOutput(
            training_tensor=logits,
            inference_tensor=inference,
            state=state if state is not None else {},
        )
