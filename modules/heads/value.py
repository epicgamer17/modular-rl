from typing import Tuple, Optional
from torch import Tensor
from .base import BaseHead
from modules.output_strategies import OutputStrategy
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

    def forward(self, x: Tensor, return_scalar: bool = True) -> Tensor:
        logits = super().forward(x)
        if return_scalar:
            return self.strategy.logits_to_scalar(logits)
        return logits
