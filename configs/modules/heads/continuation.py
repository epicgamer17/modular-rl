from configs.modules.heads.base import HeadConfig


class ContinuationHeadConfig(HeadConfig):
    """Configuration for Continuation head."""

    def __init__(self, config_dict: dict):
        super().__init__(config_dict)
