from typing import Optional
from agent_configs.base_config import ConfigBase
from configs.modules.backbones.base import BackboneConfig
from configs.modules.backbones.factory import BackboneConfigFactory


class HeadConfig(ConfigBase):
    """Base configuration for all network heads."""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        # Optional neck/backbone for the head
        self.neck: Optional[BackboneConfig] = self.parse_field(
            "neck", default=None, required=False, wrapper=BackboneConfigFactory.create
        )
        # Output Strategy configuration
        self.output_strategy: Optional[dict] = self.parse_field(
            "output_strategy", default={}, required=False
        )


class RegressionHeadConfig(HeadConfig):
    """Configuration for RegressionHead."""

    def __init__(self, config_dict: dict = None):
        super().__init__(config_dict or {})
