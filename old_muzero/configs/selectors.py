from typing import List, Optional, Dict, Any
from old_muzero.configs.base import ConfigBase


class BaseSelectorConfig(ConfigBase):
    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__(config_dict)
        self.type: str = self.parse_field("type", required=True)
        # kwargs will be passed to the selector constructor
        self.kwargs: Dict[str, Any] = self.parse_field("kwargs", {}, required=False)
        self.default_mask_value: float = self.parse_field(
            "default_mask_value", -float("inf"), required=False
        )


class DecoratorConfig(ConfigBase):
    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__(config_dict)
        self.type: str = self.parse_field("type", required=True)
        # kwargs will be passed to the decorator constructor
        self.kwargs: Dict[str, Any] = self.parse_field("kwargs", {}, required=False)


class SelectorConfig(ConfigBase):
    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__(config_dict)

        # Base selector configuration
        base_dict = self.parse_field("base", required=True)
        self.base = BaseSelectorConfig(base_dict)

        # List of decorator configurations
        decorators_list = self.parse_field("decorators", [], required=False)
        self.decorators: List[DecoratorConfig] = [
            DecoratorConfig(d) for d in decorators_list
        ]
