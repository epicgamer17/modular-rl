from typing import Tuple, Optional
from torch import Tensor
from .base import BaseHead
from modules.output_strategies import Categorical
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig


class ChanceProbabilityHead(BaseHead):
    """
    Predicts the probability distribution over chance outcomes (codes).
    Used in Stochastic MuZero.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        num_chance_codes: int,
        neck_config: Optional[BackboneConfig] = None,
    ):
        strategy = Categorical(num_classes=num_chance_codes)
        super().__init__(arch_config, input_shape, strategy, neck_config)

    def forward(self, x: Tensor, return_probs: bool = True) -> Tensor:
        logits = super().forward(x)
        if return_probs:
            return self.strategy.logits_to_probs(logits)
        return logits
