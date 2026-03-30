from .base import BackboneConfig


class DenseConfig(BackboneConfig):
    """Configuration for Dense (MLP) backbone."""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.widths: list[int] = self.parse_field("widths", [256, 256])
