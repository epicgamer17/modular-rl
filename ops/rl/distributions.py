import torch
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput

def op_log_prob(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """LogProb(logits, action) -> log_prob"""
    logits = inputs.get("logits")
    action = inputs.get("action")
    if logits is None or action is None:
        return MissingInput("logits/action")

    dist = torch.distributions.Categorical(logits=logits)
    return dist.log_prob(action.long())

def op_entropy(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """Entropy(logits) -> entropy(dist).mean()"""
    logits = inputs.get("logits")
    if logits is None:
        return MissingInput("logits")
    dist = torch.distributions.Categorical(logits=logits)
    return dist.entropy().mean()
