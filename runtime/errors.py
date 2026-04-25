from dataclasses import dataclass
from typing import Optional
from runtime.base import RuntimeValue


@dataclass(frozen=True)
class ExecutionError(RuntimeValue):
    """Indicates a failure during execution."""

    message: str
    error: Optional[Exception] = None

    def __repr__(self):
        return f"ExecutionError({self.message})"

    def __bool__(self):
        return False
