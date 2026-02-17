from .base import HeadConfig
from configs.modules.output_strategies import OutputStrategyConfigFactory


class ValueHeadConfig(HeadConfig):
    """Configuration for ValueHead."""

    def __init__(self, config_dict: dict = None):
        super().__init__(config_dict or {})
        d = config_dict or {}
        # Default to regression for Value
        strat = d.get("output_strategy", {"type": "regression"})
        self.output_strategy = OutputStrategyConfigFactory.create(strat)


class RewardHeadConfig(HeadConfig):
    """Configuration for RewardHead."""

    def __init__(self, config_dict: dict = None):
        super().__init__(config_dict or {})
        d = config_dict or {}
        # Default to regression for Reward
        strat = d.get("output_strategy", {"type": "regression"})
        self.output_strategy = OutputStrategyConfigFactory.create(strat)


class PolicyHeadConfig(HeadConfig):
    """Configuration for PolicyHead."""

    def __init__(self, config_dict: dict = None):
        super().__init__(config_dict or {})
        d = config_dict or {}
        # Default to categorical for Policy? Or just don't set if handled by head arg?
        # PolicyHead usually takes num_actions.
        # But if we want strategy (e.g. categorical), we should parse it.
        strat = d.get("output_strategy", {"type": "categorical"})
        self.output_strategy = OutputStrategyConfigFactory.create(strat)


class ChanceProbabilityHeadConfig(HeadConfig):
    """Configuration for ChanceProbabilityHead."""

    def __init__(self, config_dict: dict = None):
        super().__init__(config_dict or {})
        d = config_dict or {}
        # Default to categorical for Chance
        strat = d.get("output_strategy", {"type": "categorical"})
        self.output_strategy = OutputStrategyConfigFactory.create(strat)
