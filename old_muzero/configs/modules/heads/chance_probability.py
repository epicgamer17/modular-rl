from .base import HeadConfig
from old_muzero.configs.modules.output_strategies import OutputStrategyConfigFactory


class ChanceProbabilityHeadConfig(HeadConfig):
    """Configuration for ChanceProbabilityHead."""

    def __init__(self, config_dict: dict = None):
        super().__init__(config_dict or {})
        d = config_dict or {}
        # Default to categorical for Chance
        strat = d.get("output_strategy", {"type": "categorical"})
        self.output_strategy = OutputStrategyConfigFactory.create(strat)
