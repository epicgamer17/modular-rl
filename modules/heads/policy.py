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
    ) -> Tuple[Tensor, Dict[str, Any]]:
        logits, new_state = super().forward(x, state)
        return logits, new_state

    def get_distribution(
        self, x: Tensor, state: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.distributions.Distribution, Optional[Dict[str, Any]]]:
        logits, new_state = super().forward(x, state)  # Should we use self.forward?
        # BaseHead forward calls neck then head.
        # If we use self.forward, it calls super().forward.
        # So yes, we can call self.forward or super().forward.
        # But get_distribution usually takes observations 'x'.
        return self.strategy.get_distribution(logits), new_state
