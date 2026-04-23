"""
Operator for target network synchronization.
Supports hard and soft updates based on steps or frequency.
"""

from typing import Dict, Any, Optional
import torch
from core.graph import Node
from runtime.context import ExecutionContext

def op_target_sync(node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None) -> None:
    """
    Synchronizes parameters from a source (online) network to a target network.
    
    Parameters in node.params:
        source_net: The online network (nn.Module).
        target_net: The target network (nn.Module).
        tau: Soft update coefficient (1.0 for hard update).
        sync_type: 'periodic_hard' or 'soft'.
    """
    source_net = node.params["source_net"]
    target_net = node.params["target_net"]
    tau = node.params.get("tau", 1.0)
    sync_type = node.params.get("sync_type", "periodic_hard")

    with torch.no_grad():
        if sync_type == "periodic_hard" or tau >= 1.0:
            # Hard Update
            target_net.load_state_dict(source_net.state_dict())
        else:
            # Soft Update: target = (1 - tau) * target + tau * source
            for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + source_param.data * tau
                )
