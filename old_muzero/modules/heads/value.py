from typing import Tuple, Optional, Dict, Any
from torch import Tensor
from .base import BaseHead
from old_muzero.agents.learner.losses.representations import BaseRepresentation
from old_muzero.configs.modules.architecture_config import ArchitectureConfig
from old_muzero.configs.modules.backbones.base import BackboneConfig


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
    ) -> Tuple[Tensor, Dict[str, Any], Tensor]:
        """Returns: (logits, state, expected_value)"""
        logits, new_state = super().forward(x, state)
        expected_value = self.representation.to_expected_value(logits)
        return logits, new_state, expected_value
