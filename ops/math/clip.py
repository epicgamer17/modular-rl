import torch
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput

def op_clip(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """Clip(x, eps) -> clamp(x, 1-eps, 1+eps)"""
    x = inputs.get("x")
    eps = node.params.get("eps", 0.2)
    if x is None:
        return MissingInput("x")
    return torch.clamp(x, 1.0 - eps, 1.0 + eps)
