import torch
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput

def op_expert_actor(node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None) -> torch.Tensor:
    """ExpertActor(obs) -> actions from expert function.

    The expert is resolved via `expert_handle` from the context's CallableRegistry
    so node params stay pure (no live closures).
    """
    obs = inputs.get("obs")
    if obs is None:
        return MissingInput("obs")

    handle = node.params.get("expert_handle", "expert")
    if context is None:
        raise RuntimeError("ExpertActor requires an ExecutionContext to resolve expert_handle.")
    expert = context.get_callable(handle)

    # obs is [B, ...]
    actions = [expert(o) for o in obs]
    return torch.tensor(actions, dtype=torch.int64)
