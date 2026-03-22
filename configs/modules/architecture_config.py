from typing import Any, Callable, Optional, Dict
from configs.base import ConfigBase
from modules.utils import prepare_activations


class ArchitectureConfig(ConfigBase):
    """
    Shared configuration for network architectures, providing defaults for
    activations, normalizations, and initializers.
    """

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.activation = self.parse_field(
            "activation", "relu", wrapper=prepare_activations
        )
        self.norm_type: str = self.parse_field("norm_type", "none")
        self.noisy_sigma: float = self.parse_field("noisy_sigma", 0.0)

        self.kernel_initializer = self.parse_field(
            "kernel_initializer",
            None,
            required=False,
        )
        self.output_layer_initializer = self.parse_field(
            "output_layer_initializer",
            None,
            required=False,
        )

        # Defaults for backbone and neck (e.g., {"type": "resnet", "num_blocks": 2})
        self.backbone_defaults: Dict[str, Any] = self.parse_field(
            "backbone_defaults", {}
        )
        self.neck_defaults: Dict[str, Any] = self.parse_field("neck_defaults", {})
