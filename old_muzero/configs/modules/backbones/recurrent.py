from .base import BackboneConfig


class RecurrentConfig(BackboneConfig):
    """Configuration for Recurrent (GRU/LSTM) backbone."""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.hidden_size: int = self.parse_field("hidden_size", 256)
        self.num_layers: int = self.parse_field("num_layers", 1)
        self.rnn_type: str = self.parse_field("rnn_type", "gru")
