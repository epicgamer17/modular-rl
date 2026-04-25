import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput


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
