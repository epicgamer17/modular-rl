from typing import Tuple, Optional, Dict, Any
from torch import Tensor
import torch
from .base import BaseHead
from modules.heads.strategies import OutputStrategy, ScalarStrategy
from configs.modules.architecture_config import ArchitectureConfig
from configs.modules.backbones.base import BackboneConfig


class ObservationHead(BaseHead):
    """
    Predicts/reconstructs the observation.
    Can be used for Dreamer decoders or general autoencoding/prediction tasks.
    Flexible enough to handle image reconstruction (via neck=DeconvBackbone)
    or vector reconstruction (via neck=DenseBackbone).
    """

    def __init__(
        self,
        arch_config: ArchitectureConfig,
        input_shape: Tuple[int, ...],
        strategy: Optional[OutputStrategy] = None,
        neck_config: Optional[BackboneConfig] = None,
        use_output_layer: bool = True,
    ):
        # Pass strategy=None to avoid creating the default output layer in BaseHead if not wanted
        super().__init__(
            arch_config,
            input_shape,
            strategy if use_output_layer else None,
            neck_config,
        )
        self.use_output_layer = use_output_layer

        # Ensure we have a strategy for logic downstream
        if self.strategy is None:
            self.strategy = strategy or ScalarStrategy()

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Dict[str, Any], Tensor]:
        state = state if state is not None else {}

        # Process neck
        x = self.neck(x)

        # If we have a final output layer (dense), we project.
        if self.output_layer is not None:
            if x.dim() > 2:
                x = x.flatten(1, -1)
            logits = self.output_layer(x)
        else:
            logits = x

        # expected_observation is strategy conversion
        observation_pred = self.strategy.to_expected_value(logits)

        return logits, state, observation_pred
