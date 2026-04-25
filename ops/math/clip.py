import torch
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput
from runtime.registry import OperatorSpec, PortSpec

CLIP_SPEC = OperatorSpec.create(
    name="Clip",
    inputs={"x": PortSpec(spec=None)},
    outputs={"y": PortSpec(spec=None)},
    pure=True,
    math_category="elementwise",
    allowed_contexts={"actor", "learner"},
    differentiable=True,
    creates_grad=False,
    consumes_grad=False,
    updates_params=False,
)

def op_clip(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """Clip(x, eps) -> clamp(x, 1-eps, 1+eps)"""
    x = inputs.get("x")
    eps = node.params.get("eps", 0.2)
    if x is None:
        return MissingInput("x")
    return torch.clamp(x, 1.0 - eps, 1.0 + eps)
