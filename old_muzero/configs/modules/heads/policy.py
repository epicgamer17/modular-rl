from .base import HeadConfig
from old_muzero.configs.modules.output_strategies import OutputStrategyConfigFactory


class PolicyHeadConfig(HeadConfig):
    """Configuration for PolicyHead (Discrete or Continuous)."""

    def __init__(self, config_dict: dict = None):
        super().__init__(config_dict or {})
        d = config_dict or {}
        # Default to categorical for Policy? Or just don't set if handled by head arg?
        # PolicyHead usually takes num_actions.
        # But if we want strategy (e.g. categorical), we should parse it.
        strat = d.get("output_strategy", {"type": "categorical"})
        self.output_strategy = OutputStrategyConfigFactory.create(strat)
