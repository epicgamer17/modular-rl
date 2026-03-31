from .base import BackboneConfig


class ConvConfig(BackboneConfig):
    """Configuration for Convolutional backbone."""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.filters: list[int] = self.parse_field("filters", [32, 64, 64])
        self.kernel_sizes: list[int] = self.parse_field("kernel_sizes", [3, 3, 3])
        self.strides: list[int] = self.parse_field("strides", [2, 2, 2])
