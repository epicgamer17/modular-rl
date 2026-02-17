from typing import Tuple, Optional
from torch import Tensor
from .base import BaseHead
from modules.output_strategies import Categorical
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig


class ToPlayHead(BaseHead):
    """
    Predicts which player is currently active.
    Usually a categorical distribution over player IDs.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        num_players: int,
        neck_config: Optional[BackboneConfig] = None,
        strategy=None,
    ):
        if strategy is None:
            strategy = Categorical(num_classes=num_players)
        super().__init__(arch_config, input_shape, strategy, neck_config)

    def forward(self, x: Tensor, return_probs: bool = True) -> Tensor:
        logits = super().forward(x)
        if return_probs:
            return self.strategy.logits_to_probs(logits)
        return logits
