import torch
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput

def op_q_values_single(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """
    Computes Q-values for a single observation.
    Expects input 'obs' of shape [obs_dim].
    Returns tensor of shape [act_dim].
    """
    obs = inputs.get("obs")
    if obs is None:
        return MissingInput("obs")

    model_handle = node.params.get("model_handle", "online_q")
    q_net = context.get_model(model_handle)

    # Try to get snapshot if available (for actor consistency)
    snapshot = None
    if context and hasattr(context, "get_actor_snapshot"):
        snapshot = context.get_actor_snapshot(node.node_id)

    state = (
        snapshot.state
        if snapshot
        else {**dict(q_net.named_parameters()), **dict(q_net.named_buffers())}
    )

    with torch.inference_mode():
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)

        if obs.dim() == 1:
            q_values = functional_call(q_net, state, (obs.unsqueeze(0),))
            return q_values.squeeze(0)
        else:
            return functional_call(q_net, state, (obs,))

def op_q_forward(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """
    Computes Q-values for a batch.
    Expects input 'obs'.
    """
    obs = inputs.get("obs")
    if obs is None:
        return MissingInput("obs")

    model_handle = node.params.get("model_handle", "online_q")
    q_net = context.get_model(model_handle)
    
    if node.params.get("no_grad_region", False):
        with torch.inference_mode():
            return q_net(obs)
            
    if node.params.get("activation_checkpoint", False):
        return checkpoint(lambda x: q_net(x), obs, use_reentrant=False)
        
    return q_net(obs)

def op_gather_action_q(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """Gathers Q-values for selected actions: [B, A] -> [B]."""
    q_values = inputs.get("q_values")
    actions = inputs.get("actions")
    if q_values is None:
        return MissingInput("q_values")
    if actions is None:
        return MissingInput("actions")
    return q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)

def op_bellman_target(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """Computes standard DQN Bellman targets."""
    next_q_values = inputs.get("next_q_values")
    rewards = inputs.get("rewards")
    dones = inputs.get("dones")
    gamma = node.params.get("gamma", 0.99)
    if next_q_values is None:
        return MissingInput("next_q_values")
    if rewards is None:
        return MissingInput("rewards")
    if dones is None:
        return MissingInput("dones")
    
    with torch.no_grad():
        max_next_q = next_q_values.max(1)[0]
        
        # Enforce no-broadcast policy
        assert (
            rewards.shape == max_next_q.shape
        ), f"Rewards shape {rewards.shape} must match max_next_q shape {max_next_q.shape}"
        assert (
            dones.shape == max_next_q.shape
        ), f"Dones shape {dones.shape} must match max_next_q shape {max_next_q.shape}"

        return rewards + (1 - dones.float()) * gamma * max_next_q
