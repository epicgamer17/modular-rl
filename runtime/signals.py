from dataclasses import dataclass
from runtime.base import RuntimeValue


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
