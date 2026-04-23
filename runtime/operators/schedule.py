"""
Operators for scheduling values over time (e.g., linear decay).
"""

from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext

def op_linear_decay(node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None) -> float:
    """
    Linearly decays a value from start_val to end_val over total_steps.
    Uses context.env_step as the default clock if not provided in inputs.
    """
    start = node.params["start_val"]
    end = node.params["end_val"]
    steps = node.params["total_steps"]
    
    # Allow overriding the clock via input (e.g., from a specific node or clock handle)
    clock = inputs.get("clock")
    if clock is None:
        if context is None:
            return start
        clock = context.env_step
        
    if clock >= steps:
        return float(end)
        
    # Linear interpolation: start + (end - start) * (clock / steps)
    fraction = min(1.0, clock / steps)
    val = start + (end - start) * fraction
    return float(val)
