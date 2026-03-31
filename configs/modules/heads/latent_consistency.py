from old_muzero.configs.modules.heads.base import HeadConfig


class LatentConsistencyHeadConfig(HeadConfig):
    """Configuration for Latent Consistency head."""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
        self.projection_dim: int = self.parse_field("projection_dim", 256)
