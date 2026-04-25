import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput


def op_value_loss(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """ValueLoss(values, returns, old_values) -> 0.5 * max((v-r)^2, (v_clip-r)^2).mean()"""
    values = inputs.get("values")
    returns = inputs.get("returns")
    old_values = inputs.get("old_values")
    eps = node.params.get("eps", 0.2)
    use_clipping = node.params.get("clip", True)

    if values is None:
        return MissingInput("values")
    if returns is None:
        return MissingInput("returns")

    # Enforce no-broadcast policy
    assert (
        values.shape == returns.shape
    ), f"Values shape {values.shape} must match Returns shape {returns.shape}"
    if old_values is not None:
        assert (
            old_values.shape == values.shape
        ), f"Old Values shape {old_values.shape} must match Values shape {values.shape}"

    if use_clipping and old_values is not None:
        v_clipped = old_values + torch.clamp(values - old_values, -eps, eps)
        v_loss_unclipped = (values - returns) ** 2
        v_loss_clipped = (v_clipped - returns) ** 2
        return 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
    else:
        return 0.5 * nn.functional.mse_loss(values, returns)


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
        from torch.utils.checkpoint import checkpoint

        current_q = checkpoint(current_q_forward, states, use_reentrant=False)
    else:
        current_q = current_q_forward(states)

    with torch.no_grad():
        # target_q: [B]
        max_next_q = target_net(next_states).max(1)[0]

        # Enforce no-broadcast policy
        assert (
            rewards.shape == max_next_q.shape
        ), f"Rewards shape {rewards.shape} must match max_next_q shape {max_next_q.shape}"
        assert (
            dones.shape == max_next_q.shape
        ), f"Dones shape {dones.shape} must match max_next_q shape {max_next_q.shape}"

        target_q = rewards + (1 - dones) * gamma * max_next_q

    # Enforce no-broadcast in MSE loss input
    assert (
        current_q.shape == target_q.shape
    ), f"Current Q shape {current_q.shape} must match Target Q shape {target_q.shape}"
    return nn.functional.mse_loss(current_q, target_q)
