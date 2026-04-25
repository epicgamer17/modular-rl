import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput
from runtime.registry import OperatorSpec, PortSpec, Scalar

MSE_LOSS_SPEC = OperatorSpec.create(
    name="MSELoss",
    inputs={"pred": PortSpec(spec=None), "target": PortSpec(spec=None)},
    outputs={"loss": Scalar("float32")},
    differentiable=True,
    creates_grad=True,
    consumes_grad=False,
    updates_params=False,
    allowed_contexts={"learner"},
)


def op_mse_loss(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    pred = inputs.get("pred")
    target = inputs.get("target")
    if pred is None:
        return MissingInput("pred")
    if target is None:
        return MissingInput("target")
    return nn.functional.mse_loss(pred, target)
