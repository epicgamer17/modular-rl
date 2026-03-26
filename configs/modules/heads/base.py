from typing import Optional
from configs.base import ConfigBase
from configs.modules.backbones.base import BackboneConfig


class HeadConfig(ConfigBase):
    """Base configuration for all network heads."""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        # Optional neck/backbone for the head
        from agents.factories.backbone_config import BackboneConfigFactory
        self.neck: Optional[BackboneConfig] = self.parse_field(
            "neck", default=None, required=False, wrapper=BackboneConfigFactory.create
        )
        # Input Source configuration
        self.input_source: str = self.parse_field(
            "input_source", default="default", required=False
        )
        # Output Strategy configuration
        self.output_strategy: Optional[dict] = self.parse_field(
            "output_strategy", default={}, required=False
        )
