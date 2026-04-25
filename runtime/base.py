class RuntimeValue:
    """Base class for all explicit runtime values in the RL IR."""

    @property
    def has_data(self) -> bool:
        """Returns True if this value contains actual data payload."""
        return False

    def __bool__(self):
        # By default, RuntimeValues are truthy except for specific ones
        return False
