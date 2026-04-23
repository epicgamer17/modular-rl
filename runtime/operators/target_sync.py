"""
Operator for target network synchronization.
Supports hard and soft updates based on steps or frequency.
"""

from typing import Dict, Any, Optional
import torch
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.values import NoOp, Skipped

def op_target_sync(node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None) -> None:
    """
    Synchronizes parameters from a source (online) network to a target network.
    
    If any input is None, skipping synchronization (assumes dependent update skipped).
    """
    if inputs and any(v is None for v in inputs.values()):
        return Skipped("missing_inputs")
    source_handle = node.params["source_handle"]
    target_handle = node.params["target_handle"]
    
    if not context:
        raise RuntimeError("TargetSync requires an ExecutionContext to resolve model handles.")
        
    source_net = context.get_model(source_handle)
    target_net = context.get_model(target_handle)
    tau = node.params.get("tau", 1.0)
    sync_type = node.params.get("sync_type", "periodic_hard")
    freq = node.params.get("sync_frequency", 1)
    sync_on = node.params.get("sync_on", "learner_step")

    # 1. Frequency Check
    if context and freq > 1:
        clock = context.env_step if sync_on == "env_step" else context.learner_step
        if clock % freq != 0:
            return NoOp()

    with torch.no_grad():
        if sync_type == "periodic_hard" or tau >= 1.0:
            # Hard Update: target = source
            target_net.load_state_dict(source_net.state_dict())
        else:
            # Soft Update: target = tau * source + (1 - tau) * target
            for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + source_param.data * tau
                )
    return NoOp()
