from typing import Tuple, Optional, Dict, Any
from torch import Tensor
from .base import BaseHead
from modules.heads.strategies import Categorical
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

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        logits, new_state = super().forward(x, state)
        return logits, new_state
