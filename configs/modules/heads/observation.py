from configs.modules.heads.base import HeadConfig


class ObservationHeadConfig(HeadConfig):
    """Configuration for Observation head."""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
