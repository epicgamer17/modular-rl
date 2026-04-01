from typing import Optional
from configs.base import ConfigBase


class OutputStrategyConfig(ConfigBase):
    """Base configuration for output strategies."""

    pass


class ScalarStrategyConfig(OutputStrategyConfig):
    def __init__(self, config_dict: dict = None):
        super().__init__(config_dict or {})


class CategoricalConfig(OutputStrategyConfig):
    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.num_classes: int = self.parse_field("num_classes", required=True)


class GaussianConfig(OutputStrategyConfig):
    def __init__(self, config_dict: dict = None):
        super().__init__(config_dict or {})
        d = config_dict or {}
        self.action_dim = d.get("action_dim")
        self.min_log_std = d.get("min_log_std", -20.0)
        self.max_log_std = d.get("max_log_std", 2.0)



class C51SupportConfig(OutputStrategyConfig):
    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.v_min: float = self.parse_field("v_min", -10.0)
        self.v_max: float = self.parse_field("v_max", 10.0)
        self.num_atoms: int = self.parse_field("num_atoms", 51)


class OutputStrategyConfigFactory:
    @staticmethod
    def create(config_dict: dict) -> OutputStrategyConfig:
        if config_dict is None:
            return ScalarStrategyConfig({})

        strategy_type = config_dict.get("type", "scalar")

        if strategy_type == "scalar":
            return ScalarStrategyConfig(config_dict)
        elif strategy_type == "categorical":
            return CategoricalConfig(config_dict)
        elif strategy_type == "gaussian" or strategy_type == "continuous":
            return GaussianConfig(config_dict)
        elif strategy_type == "c51":
            return C51SupportConfig(config_dict)
        else:
            return ScalarStrategyConfig(config_dict)
