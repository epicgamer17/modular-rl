from .base import HeadConfig
from configs.modules.output_strategies import OutputStrategyConfigFactory


class RewardHeadConfig(HeadConfig):
    """Configuration for RewardHead."""

    def __init__(self, config_dict: dict = None):
        super().__init__(config_dict or {})
        d = config_dict or {}
        # Default to regression for Reward
        strat = d.get("output_strategy", {"type": "scalar"})
        self.output_strategy = OutputStrategyConfigFactory.create(strat)


class ValuePrefixRewardHeadConfig(RewardHeadConfig):
    """Configuration for ValuePrefixRewardHead (LSTM-based)."""

    def __init__(self, config_dict: dict = None):
        super().__init__(config_dict)
        d = config_dict or {}
        self.lstm_hidden_size: int = d.get("lstm_hidden_size", 64)
        self.lstm_horizon_len: int = d.get("lstm_horizon_len", 5)
