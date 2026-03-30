from .base import BackboneConfig


class IdentityConfig(BackboneConfig):
    """Configuration for IdentityBackbone (pass-through)."""

    def __init__(self, config_dict: dict = None):
        super().__init__(config_dict or {})
        # Identity backbone has no parameters usually.
