import torch
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput

def op_weighted_sum(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """WeightedSum(**inputs) -> sum(weight * tensor)"""
    tensors = []
    for name, tensor in inputs.items():
        if isinstance(tensor, (torch.Tensor, float, int)):
            weight = node.params.get(name, 1.0)
            tensors.append(weight * tensor)
    
    if not tensors:
        return torch.tensor(0.0)
        
    total = tensors[0]
    for t in tensors[1:]:
        total = total + t
        
    return total

def op_reduce_mean(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """Computes mean of input tensor."""
    val = inputs.get("input")
    if val is None:
        return MissingInput("input")
    return val.mean()
