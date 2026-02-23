from typing import Tuple, Optional, Dict, Any
from torch import Tensor
from .base import BaseHead
from modules.heads.strategies import OutputStrategy, ScalarStrategy
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig
from modules.blocks.dense import build_dense


class LatentConsistencyHead(BaseHead):
    """
    Projects latent states into a embedding space for consistency loss.
    Commonly used in MuZero (projection head) or Dreamer.
    Typically a small MLP.
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        strategy: Optional[OutputStrategy] = None,
        neck_config: Optional[BackboneConfig] = None,
        projection_dim: int = 256,
    ):
        # If strategy is None, default to ScalarStrategy for embedding identity
        if strategy is None:
            strategy = ScalarStrategy(projection_dim)

        super().__init__(arch_config, input_shape, strategy, neck_config)
        self.projection_dim = projection_dim

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any], Tensor]:
        state = state if state is not None else {}

        # Standard neck -> output_layer
        logits, _ = super().forward(x, state)

        # The projected latent
        projected_latent = self.strategy.to_expected_value(logits)

        return logits, state, projected_latent
