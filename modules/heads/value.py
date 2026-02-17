from typing import Tuple, Optional, Dict, Any
from torch import Tensor
from .base import BaseHead
from modules.heads.strategies import OutputStrategy
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig


class ValueHead(BaseHead):
    """
    Predicts the expected return (Value).
    Supports multiple output strategies (Regression, MuZero, C51, Dreamer).
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        strategy: OutputStrategy,
        neck_config: Optional[BackboneConfig] = None,
    ):
        super().__init__(arch_config, input_shape, strategy, neck_config)

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        logits, new_state = super().forward(x, state)
        return logits, new_state
