from typing import Optional
from agent_configs.base_config import ConfigBase


class OutputStrategyConfig(ConfigBase):
    """Base configuration for output strategies."""

    pass


class RegressionConfig(OutputStrategyConfig):
    def __init__(self, config_dict: dict = None):
        super().__init__(config_dict or {})


class CategoricalConfig(OutputStrategyConfig):
    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.num_classes: int = self.parse_field("num_classes", required=True)


class MuZeroSupportConfig(OutputStrategyConfig):
    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.support_range: int = self.parse_field("support_range", 10)
        self.eps: float = self.parse_field("eps", 0.001)


class C51SupportConfig(OutputStrategyConfig):
    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.v_min: float = self.parse_field("v_min", -10.0)
        self.v_max: float = self.parse_field("v_max", 10.0)
        self.num_atoms: int = self.parse_field("num_atoms", 51)


class DreamerSupportConfig(OutputStrategyConfig):
    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.support_range: int = self.parse_field("support_range", 20)


class OutputStrategyConfigFactory:
    @staticmethod
    def create(config_dict: dict) -> OutputStrategyConfig:
        if config_dict is None:
            return RegressionConfig({})

        strategy_type = config_dict.get("type", "regression")

        if strategy_type == "regression":
            return RegressionConfig(config_dict)
        elif strategy_type == "categorical":
            return CategoricalConfig(config_dict)
        elif strategy_type == "muzero":
            return MuZeroSupportConfig(config_dict)
        elif strategy_type == "c51":
            return C51SupportConfig(config_dict)
        elif strategy_type == "dreamer":
            return DreamerSupportConfig(config_dict)
        else:
            return RegressionConfig(config_dict)
