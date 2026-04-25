from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput

def op_get_field(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> Any:
    """Extracts a field from a dictionary or object."""
    val = inputs.get("input")
    field = node.params.get("field")
    if val is None:
        return MissingInput("input")
    if isinstance(val, dict):
        return val.get(field)
    return getattr(val, field)
