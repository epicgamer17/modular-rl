from .base import HeadConfig
from configs.modules.output_strategies import (
    CategoricalConfig,
    OutputStrategyConfigFactory,
)


class ToPlayHeadConfig(HeadConfig):
    """Configuration for ToPlayHead."""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        # Note: num_players usually comes from game config,
        # but can be specified here if needed or passed dynamically.
        self.num_players: Optional[int] = self.parse_field(
            "num_players", None, required=False
        )
        strat = config_dict.get("output_strategy", {"type": "categorical"})
        self.output_strategy = OutputStrategyConfigFactory.create(strat)
