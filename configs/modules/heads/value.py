from .base import HeadConfig
from configs.modules.output_strategies import OutputStrategyConfigFactory


class ValueHeadConfig(HeadConfig):
    """Configuration for ValueHead."""

    def __init__(self, config_dict: dict = None):
        super().__init__(config_dict or {})
        d = config_dict or {}
        # Default to regression for Value
        strat = d.get("output_strategy", {"type": "scalar"})
        self.output_strategy = OutputStrategyConfigFactory.create(strat)
