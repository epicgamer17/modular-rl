

from configs.modules.heads.base import HeadConfig


class SimSiamProjectorConfig(HeadConfig):
    """Configuration for SimSiam/BYOL Projection Head."""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.proj_hidden_dim: int = self.parse_field("proj_hidden_dim", 2048)
        self.proj_output_dim: int = self.parse_field("proj_output_dim", 2048)
        self.pred_hidden_dim: int = self.parse_field("pred_hidden_dim", 512)
        self.pred_output_dim: int = self.parse_field("pred_output_dim", 2048)
