import torch
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput
from runtime.registry import OperatorSpec, TensorSpec

LOG_PROB_SPEC = OperatorSpec.create(
    name="LogProb",
    inputs={
        "logits": TensorSpec(shape=(-1, -1), dtype="float32"),
        "action": TensorSpec(shape=(-1,), dtype="int64"),
    },
    outputs={"log_prob": TensorSpec(shape=(-1,), dtype="float32")},
    domain_tags={"policy_gradient"},
    math_category="distribution",
    allowed_contexts={"actor", "learner"},
    differentiable=True,
    creates_grad=True,
    consumes_grad=False,
    updates_params=False,
)

ENTROPY_SPEC = OperatorSpec.create(
    name="Entropy",
    inputs={
        "logits": TensorSpec(shape=(-1, -1), dtype="float32"),
    },
    outputs={"entropy": TensorSpec(shape=(), dtype="float32")},
    domain_tags={"policy_gradient"},
    math_category="reduction",
    allowed_contexts={"actor", "learner"},
    differentiable=True,
    creates_grad=True,
    consumes_grad=False,
    updates_params=False,
)

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
