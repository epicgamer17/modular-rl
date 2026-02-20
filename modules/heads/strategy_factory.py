from typing import Optional, Dict, Type
from torch import nn
from configs.modules.output_strategies import (
    OutputStrategyConfig,
    RegressionConfig,
    CategoricalConfig,
    MuZeroSupportConfig,
    C51SupportConfig,
    GaussianConfig,
)
from modules.heads.strategies import (
    OutputStrategy,
    Regression,
    Categorical,
    MuZeroSupport,
    C51Support,
    GaussianStrategy,
)


class OutputStrategyFactory:
    """Factory to create OutputStrategy modules based on their configuration."""

    _strategies: Dict[Type[OutputStrategyConfig], Type[OutputStrategy]] = {
        RegressionConfig: Regression,
        CategoricalConfig: Categorical,
        MuZeroSupportConfig: MuZeroSupport,
        C51SupportConfig: C51Support,
        GaussianConfig: GaussianStrategy,
    }

    @classmethod
    def create(cls, config: OutputStrategyConfig) -> OutputStrategy:
        config_type = type(config)
        if config_type not in cls._strategies:
            # Default to Regression if unknown
            return Regression()

        # Extract arguments from config
        if isinstance(config, CategoricalConfig):
            return Categorical(num_classes=config.num_classes)
        elif isinstance(config, MuZeroSupportConfig):
            return MuZeroSupport(support_range=config.support_range, eps=config.eps)
        elif isinstance(config, GaussianConfig):
            return GaussianStrategy(
                action_dim=config.action_dim if config.action_dim else 0,
                min_log_std=config.min_log_std,
                max_log_std=config.max_log_std,
            )
        elif isinstance(config, C51SupportConfig):
            return C51Support(
                v_min=config.v_min, v_max=config.v_max, num_atoms=config.num_atoms
            )

        return cls._strategies[config_type]()
