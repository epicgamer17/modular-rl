from .base import BackboneConfig


class TransformerConfig(BackboneConfig):
    """Configuration for Transformer backbone."""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.num_layers: int = self.parse_field("num_layers", 4)
        self.num_heads: int = self.parse_field("num_heads", 8)
        self.d_model: int = self.parse_field("d_model", 256)
        self.d_ff: int = self.parse_field("d_ff", 1024)
        self.dropout: float = self.parse_field("dropout", 0.1)
