import torch
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import NoOp, Skipped
from runtime.registry import OperatorSpec

TARGET_SYNC_SPEC = OperatorSpec.create(
    "TargetSync",
    inputs={},
    outputs={},
    pure=False,
    stateful=True,
    allowed_contexts={"learner"},
    side_effects=["target_update"],
    requires_models=["source", "target"],
    math_category="control",
    differentiable=False,
    creates_grad=False,
    consumes_grad=False,
    updates_params=False,
)

def op_target_sync(node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None) -> None:
    """
    Synchronizes parameters from a source (online) network to a target network.
    """
    if inputs and any(v is None for v in inputs.values()):
        return Skipped("missing_inputs")
    model_handle = node.params["model_handle"]
    target_handle = node.params["target_handle"]
    
    if not context:
        raise RuntimeError("TargetSync requires an ExecutionContext to resolve model handles.")
        
    source_net = context.get_model(model_handle)
    target_net = context.get_model(target_handle)
    tau = node.params.get("tau", 1.0)
    sync_type = node.params.get("sync_type", "periodic_hard")
    freq = node.params.get("sync_frequency", 1)
    sync_on = node.params.get("sync_on", "learner_step")

    if context and freq > 1:
        clock = context.env_step if sync_on == "env_step" else context.learner_step
        if clock % freq != 0:
            return NoOp()

    with torch.no_grad():
        if sync_type == "periodic_hard" or tau >= 1.0:
            target_net.load_state_dict(source_net.state_dict())
        else:
            for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + source_param.data * tau
                )
    return NoOp()
