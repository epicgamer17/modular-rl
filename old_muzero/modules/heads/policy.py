from typing import Tuple, Optional, Dict, Any
import torch
from torch import Tensor
from .base import BaseHead
from old_muzero.agents.learner.losses.representations import BaseRepresentation
from old_muzero.configs.modules.architecture_config import ArchitectureConfig
from old_muzero.configs.modules.backbones.base import BackboneConfig


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
    ) -> Tuple[Tensor, Dict[str, Any], Any]:
        """Returns: (logits, state, inference)"""
        logits, new_state = super().forward(x, state)
        inference = self.representation.to_inference(logits)
        return logits, new_state, inference
