from typing import Tuple, Optional
import torch
from torch import Tensor
from .base import BaseHead
from modules.output_strategies import Categorical
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig


class PolicyHead(BaseHead):
    """
    Predicts the action distribution (Policy).
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        num_actions: int,
        neck_config: Optional[BackboneConfig] = None,
        strategy=None,
    ):
        if strategy is None:
            strategy = Categorical(num_classes=num_actions)
        super().__init__(arch_config, input_shape, strategy, neck_config)

    def forward(self, x: Tensor, return_probs: bool = True) -> Tensor:
        logits = super().forward(x)
        if return_probs:
            return self.strategy.logits_to_probs(logits)
        return logits

    def get_distribution(self, x: Tensor) -> torch.distributions.Distribution:
        logits = super().forward(x)
        return torch.distributions.Categorical(logits=logits)
