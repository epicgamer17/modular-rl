from typing import Tuple, Optional, Dict, Any
from torch import Tensor
from .base import BaseHead
from modules.heads.strategies import OutputStrategy, Categorical, ScalarStrategy
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig


class ContinuationHead(BaseHead):
    """
    Predicts if the episode should continue (1.0) or end (0.0).
    Commonly used in Dreamer (gamma predictor) or general RL for termination prediction.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        strategy: Optional[OutputStrategy] = None,
        neck_config: Optional[BackboneConfig] = None,
    ):
        # Default to ScalarStrategy(1) if none provided, but often used as Categorical(2)
        if strategy is None:
            strategy = ScalarStrategy(1)

        super().__init__(arch_config, input_shape, strategy, neck_config)

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any], Tensor]:
        state = state if state is not None else {}
        logits, _ = super().forward(x, state)

        # continuation is essentially the expected value (probability if categorical)
        continuation = self.strategy.to_expected_value(logits)

        # If it's categorical(2), to_expected_value might return argmax.
        # For continuation, we might want the probability of class 1 if it's categorical.
        if isinstance(self.strategy, Categorical) and self.strategy.num_bins == 2:
            import torch.nn.functional as F

            probs = F.softmax(logits, dim=-1)
            continuation = probs[..., 1]  # Probability of "continue"

        return logits, state, continuation
