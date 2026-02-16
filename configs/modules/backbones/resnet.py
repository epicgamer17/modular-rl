from .base import BackboneConfig


class ResNetConfig(BackboneConfig):
    """Configuration for ResNet backbone."""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.filters: list[int] = self.parse_field("filters", [256] * 20)
        self.kernel_sizes: list[int] = self.parse_field("kernel_sizes", [3] * 20)
        self.strides: list[int] = self.parse_field("strides", [1] * 20)
