from typing import Tuple, Optional, Dict, Any
import torch
from torch import Tensor
from .base import BaseHead
from modules.heads.strategies import OutputStrategy
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig


class PolicyHead(BaseHead):
    """
    Predicts the action distribution (Policy).
    Supports both discrete (Categorical) and continuous (Gaussian) actions via OutputStrategy.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        neck_config: Optional[BackboneConfig] = None,
        strategy: Optional[OutputStrategy] = None,
    ):
        super().__init__(arch_config, input_shape, strategy, neck_config)

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any], Optional[torch.distributions.Distribution]]:
        logits, new_state = super().forward(x, state)
        dist = None
        if self.strategy is not None:
            dist = self.strategy.get_distribution(logits)
        return logits, new_state, dist
