from typing import List, Optional
from .base import HeadConfig
from old_muzero.configs.modules.output_strategies import (
    OutputStrategyConfig,
    ScalarStrategyConfig,
    OutputStrategyConfigFactory,
)


class QHeadConfig(HeadConfig):
    """
    Configuration for a standard Q-Network Head.
    Input -> Hidden Layers -> Output (Action Values or Distribution)
    """

    def __init__(self, config_dict: dict = None):
        super().__init__(config_dict or {})
        self.output_strategy: OutputStrategyConfig = self.parse_output_strategy(
            config_dict
        )
        self.hidden_widths: List[int] = self.parse_field(
            "hidden_widths", [512], required=False
        )
        self.noisy_sigma: float = self.parse_field("noisy_sigma", 0.5, required=False)

    def parse_output_strategy(self, config_dict: dict) -> OutputStrategyConfig:
        # Check if output_strategy is in config_dict
        strategy_dict = config_dict.get("output_strategy")
        return OutputStrategyConfigFactory.create(strategy_dict)


class DuelingQHeadConfig(HeadConfig):
    """
    Configuration for a Dueling Q-Network Head.
    Input -> [Value Stream, Advantage Stream] -> Aggregation -> Output
    """

    def __init__(self, config_dict: dict = None):
        super().__init__(config_dict or {})
        self.output_strategy: OutputStrategyConfig = self.parse_output_strategy(
            config_dict
        )
        self.value_hidden_widths: List[int] = self.parse_field(
            "value_hidden_widths", [512], required=False
        )
        self.advantage_hidden_widths: List[int] = self.parse_field(
            "advantage_hidden_widths", [512], required=False
        )
        self.noisy_sigma: float = self.parse_field("noisy_sigma", 0.5, required=False)

    def parse_output_strategy(self, config_dict: dict) -> OutputStrategyConfig:
        strategy_dict = config_dict.get("output_strategy")
        return OutputStrategyConfigFactory.create(strategy_dict)
