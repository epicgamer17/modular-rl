from .base import BackboneConfig


class MLPResNetConfig(BackboneConfig):
    """Configuration for MLPResNet backbone (MLP with skip connections)."""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.widths: list[int] = self.parse_field("widths", [256] * 10)
