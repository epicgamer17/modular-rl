import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput


def op_cross_entropy_loss(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """Generic cross entropy loss."""
    logits = inputs.get("logits")
    target = inputs.get("target")
    if logits is None or target is None:
        return MissingInput("logits/target")
    return nn.functional.cross_entropy(logits, target.long())


def op_sl_policy_loss(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """Supervised Learning loss for a policy (cross-entropy)."""
    batch = inputs.get("batch")
    if batch is None:
        return MissingInput("batch")

    if isinstance(batch, dict):
        obs = batch.get("obs")
        action = batch.get("action")
    else:
        obs = getattr(batch, "obs", None)
        action = getattr(batch, "action", None)
    if obs is None or action is None:
        return MissingInput("batch")

    model_handle = node.params.get("model_handle", "policy")
    policy_net = context.get_model(model_handle)

    actions = action.long()
    output = policy_net(obs)
    if isinstance(output, torch.Tensor):
        return nn.functional.cross_entropy(output, actions)
    else:
        logits = output[0]
        return nn.functional.cross_entropy(logits, actions)
