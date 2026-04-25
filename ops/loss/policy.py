import torch
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput


def op_surrogate_loss(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """SurrogateLoss(ratio, advantages, clipped_ratio) -> -min(ratio*A, clipped*A).mean()"""
    ratio = inputs.get("ratio")
    clipped_ratio = inputs.get("clipped_ratio")
    advantages = inputs.get("advantages")
    if ratio is None:
        return MissingInput("ratio")
    if clipped_ratio is None:
        return MissingInput("clipped_ratio")
    if advantages is None:
        return MissingInput("advantages")

    # Enforce no-broadcast policy
    assert (
        ratio.shape == advantages.shape
    ), f"Ratio shape {ratio.shape} must match Advantages shape {advantages.shape}"
    assert (
        clipped_ratio.shape == advantages.shape
    ), f"Clipped Ratio shape {clipped_ratio.shape} must match Advantages shape {advantages.shape}"

    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    return -torch.min(surr1, surr2).mean()


def op_entropy_loss(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """EntropyLoss(logits) -> -ent_coeff * entropy(dist).mean()"""
    logits = inputs.get("logits")
    if logits is None:
        return MissingInput("logits")

    # Optional entropy coefficient
    ent_coeff = node.params.get("ent_coeff", 1.0)

    dist = torch.distributions.Categorical(logits=logits)
    # We return the loss (negative entropy)
    return -ent_coeff * dist.entropy().mean()
