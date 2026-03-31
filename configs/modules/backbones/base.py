from typing import Any, Callable
from configs.base import ConfigBase
from modules.utils import prepare_activations
from torch import nn


class BackboneConfig(ConfigBase):
    """Base configuration for all network backbones."""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.activation = self.parse_field(
            "activation", "relu", wrapper=prepare_activations
        )
        self.norm_type: str = self.parse_field("norm_type", "none")
        self.noisy_sigma: float = self.parse_field("noisy_sigma", 0.0)
