from typing import Tuple, Optional, Dict, Any
import torch
from torch import Tensor
from .base import BaseHead
from modules.output_strategies import OutputStrategy
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
        # Legacy/Helper args
        num_actions: Optional[int] = None,
        output_size: Optional[int] = None,
    ):
        if strategy is None:
            # Lazy import to avoid circular dependencies if any
            from modules.output_strategy_factory import OutputStrategyFactory
            from configs.modules.output_strategies import (
                CategoricalConfig,
                GaussianConfig,
            )

            # Legacy Discrete support
            if num_actions is not None:
                config = CategoricalConfig(
                    {"num_classes": num_actions, "type": "categorical"}
                )
                strategy = OutputStrategyFactory.create(config)
            elif output_size is not None:
                # Assume continuous/gaussian if output_size passed and no strategy
                config = GaussianConfig({"action_dim": output_size, "type": "gaussian"})
                strategy = OutputStrategyFactory.create(config)

        super().__init__(arch_config, input_shape, strategy, neck_config)

    def forward(
        self,
        x: Tensor,
        state: Optional[Dict[str, Any]] = None,
        return_probs: bool = False,  # Deprecated argument, kept for compatibility if needed but get_distribution preferred
    ) -> Tuple[Tensor, Dict[str, Any]]:
        logits, new_state = super().forward(x, state)
        if return_probs:
            # This might break given different strategies, but for categorical it works.
            # Ideally we remove this usage.
            pass
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
