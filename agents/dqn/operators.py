import torch
import torch.nn as nn
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.values import MissingInput, NoOp
from runtime.executor import register_operator


def op_q_values_single(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """
    Computes Q-values for a single observation.
    Expects input 'obs' of shape [obs_dim] or a dict containing 'obs'.
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
        # TODO: try to remove this from ops and make it a contract or something?
        # Ensure obs is a tensor
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)

        # Ensure obs has batch dimension for the model, then squeeze it back
        # obs: [D] -> [1, D] -> model -> [1, A] -> [A]
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


def op_td_loss(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """
    Computes MSE TD Loss for DQN.
    Expects input 'batch' with 'obs', 'action', 'reward', 'next_obs', 'done'.
    """
    batch = inputs.get("batch")
    if batch is None:
        return MissingInput("batch")

    model_handle = node.params.get("model_handle", "online_q")
    target_handle = node.params.get("target_handle", "target_q")
    gamma = node.params.get("gamma", 0.99)

    q_net = context.get_model(model_handle)
    target_net = context.get_model(target_handle)

    states = batch.obs
    actions = batch.action.long()
    rewards = batch.reward
    next_states = batch.next_obs
    dones = batch.done

    def current_q_forward(x: torch.Tensor) -> torch.Tensor:
        return q_net(x).gather(1, actions.unsqueeze(1)).squeeze(1)

    if node.params.get("activation_checkpoint", False):
        current_q = checkpoint(current_q_forward, states, use_reentrant=False)
    else:
        current_q = current_q_forward(states)

    with torch.no_grad():
        # target_q: [B]
        max_next_q = target_net(next_states).max(1)[0]
        
        # Enforce no-broadcast policy
        assert rewards.shape == max_next_q.shape, f"Rewards shape {rewards.shape} must match max_next_q shape {max_next_q.shape}"
        assert dones.shape == max_next_q.shape, f"Dones shape {dones.shape} must match max_next_q shape {max_next_q.shape}"
        
        target_q = rewards + (1 - dones) * gamma * max_next_q

    # Enforce no-broadcast in MSE loss input
    assert current_q.shape == target_q.shape, f"Current Q shape {current_q.shape} must match Target Q shape {target_q.shape}"
    return nn.functional.mse_loss(current_q, target_q)


def op_optimizer_step(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> float:
    """
    Performs an optimization step.
    Expects input 'loss'.
    """
    loss = inputs.get("loss")
    if loss is None:
        return MissingInput("loss")

    # Resolve optimizer via handle from context (clean architecture)
    opt_handle = node.params.get("optimizer_handle", "main_opt")
    if context:
        opt_state = context.get_optimizer(opt_handle)
        opt_state.step(loss)

    return loss.item()


def op_reduce_mean(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """Computes mean of input tensor."""
    val = inputs.get("input")
    if val is None:
        return MissingInput("input")
    return val.mean()


def op_get_field(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> Any:
    """Extracts a field from a dictionary or object."""
    val = inputs.get("input")
    field = node.params.get("field")
    if val is None:
        return MissingInput("input")
    if isinstance(val, dict):
        return val.get(field)
    return getattr(val, field)


def op_gather_action_q(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
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
        assert rewards.shape == max_next_q.shape, f"Rewards shape {rewards.shape} must match max_next_q shape {max_next_q.shape}"
        assert dones.shape == max_next_q.shape, f"Dones shape {dones.shape} must match max_next_q shape {max_next_q.shape}"
        
        return rewards + (1 - dones.float()) * gamma * max_next_q


def register_dqn_operators():
    """Register all DQN related operators."""
    register_operator("QValuesSingle", op_q_values_single)
    register_operator("QForward", op_q_forward)
    register_operator("TDLoss", op_td_loss)
    register_operator("Optimizer", op_optimizer_step)
    register_operator("ReduceMean", op_reduce_mean)
    register_operator("GetField", op_get_field)
    register_operator("GatherActionQ", op_gather_action_q)
    register_operator("BellmanTarget", op_bellman_target)
