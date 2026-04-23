from dataclasses import dataclass
from typing import Any, Optional

class RuntimeValue:
    """Base class for all explicit runtime values in the RL IR."""
    @property
    def has_data(self) -> bool:
        """Returns True if this value contains actual data payload."""
        return False

    def __bool__(self):
        # By default, RuntimeValues are truthy except for specific ones
        return False

@dataclass(frozen=True)
class Value(RuntimeValue):
    """A standard data payload."""
    data: Any
    
    @property
    def has_data(self) -> bool:
        return True

    def __repr__(self):
        return f"Value({self.data})"
    
    def __bool__(self):
        return True

class NoOp(RuntimeValue):
    """Indicates the operator completed successfully but produced no side effects or data."""
    def __repr__(self):
        return "NoOp"
    
    def __bool__(self):
        return False

@dataclass(frozen=True)
class Skipped(RuntimeValue):
    """Indicates the operator was intentionally skipped (e.g. min_size not reached)."""
    reason: str
    
    def __repr__(self):
        return f"Skipped({self.reason})"
    
    def __bool__(self):
        return False

@dataclass(frozen=True)
class MissingInput(RuntimeValue):
    """Indicates a required input was missing or itself skipped/noop."""
    input_name: str
    
    def __repr__(self):
        return f"MissingInput({self.input_name})"
    
    def __bool__(self):
        return False

@dataclass(frozen=True)
class ExecutionError(RuntimeValue):
    """Indicates a failure during execution."""
    message: str
    error: Optional[Exception] = None
    
    def __repr__(self):
        return f"ExecutionError({self.message})"
    
    def __bool__(self):
        return False
