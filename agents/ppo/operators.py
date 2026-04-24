import torch
import torch.nn as nn
from torch.func import functional_call
from typing import Dict, Any, Optional

from core.graph import Node
from runtime.executor import register_operator
from runtime.values import MissingInput
from runtime.context import ExecutionContext


def op_policy_actor(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
):
    """
    Operator for PPO policy actor.

    Args:
        node: The graph node.
        inputs: Input values (must contain 'obs').
        context: Execution context for model access.

    Returns:
        Dictionary containing 'action', 'log_prob', and 'policy_version'.
    """
    obs = inputs.get("obs")
    if obs is None:
        return MissingInput("obs")

    model_handle = node.params.get("model_handle", "ppo_net")
    ac_net = context.get_model(model_handle)

    # Snapshot binding is handled by ActorRuntime automatically via ExecutionContext
    snapshot = context.get_actor_snapshot(node.node_id) if context else None

    if snapshot:
        params = snapshot.parameters
        version = snapshot.policy_version
    else:
        params = dict(ac_net.named_parameters())
        version = 0

    with torch.inference_mode():
        # Pass both parameters and buffers (PPO AC net might have buffers)
        probs, value = functional_call(ac_net, params, (obs.unsqueeze(0),))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

    return {
        "action": action.item(),
        "log_prob": log_prob.item(),
        "value": value.item(),
        "policy_version": version,
    }


def op_gae(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
):
    """
    Generalized Advantage Estimation (GAE) operator.
    Correctly handles termination vs truncation (timeout masking).

    Args:
        node: The graph node.
        inputs: Input values (must contain 'batch').
        context: Execution context for model access.

    Returns:
        Dictionary containing 'advantages' and 'returns'.
    """
    batch = inputs.get("batch")
    if batch is None:
        return MissingInput("batch")

    gamma = node.params["gamma"]
    gae_lambda = node.params["gae_lambda"]
    model_handle = node.params.get("model_handle", "ppo_net")
    ac_net = context.get_model(model_handle)

    obs = batch["obs"]
    rewards = batch["reward"]
    # We now expect terminated and truncated separately
    terminateds = batch["terminated"].float()
    truncateds = batch["truncated"].float()
    next_obs = batch["next_obs"]

    with torch.no_grad():
        _, values = ac_net(obs)
        # Bootstrap value for the very last state
        _, next_values_last = ac_net(next_obs[-1].unsqueeze(0))
        # We need the values for ALL states to do the shifted calculation
        # Or we can do it step-by-step
        values = values.view(-1)
        next_values_last = next_values_last.view(-1)

    advantages = torch.zeros_like(rewards)
    last_gae = 0

    # We need next_values for each step
    # For on-policy rollouts, next_obs[t] is the state after obs[t]
    # So we can compute next_values for all t
    with torch.no_grad():
        _, next_values_all = ac_net(next_obs)
        next_values_all = next_values_all.view(-1)

    for t in reversed(range(len(rewards))):
        # Timeout masking: only reset GAE on actual termination
        # non_terminal = 1 if NOT terminated
        non_terminal = 1.0 - terminateds[t]

        # delta_t = r_t + gamma * V(s_{t+1}) * non_terminal - V(s_t)
        delta = rewards[t] + gamma * next_values_all[t] * non_terminal - values[t]

        # A_t = delta_t + gamma * lam * non_terminal * A_{t+1}
        advantages[t] = last_gae = delta + gamma * gae_lambda * non_terminal * last_gae

    returns = advantages + values
    return {"advantages": advantages, "returns": returns}


def op_ppo_objective(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
):
    """
    PPO Objective (loss) operator.

    Args:
        node: The graph node.
        inputs: Input values (must contain 'batch' and 'gae').
        context: Execution context for model access.

    Returns:
        The total PPO loss.
    """
    batch = inputs.get("batch")
    gae_data = inputs.get("gae")
    if batch is None or gae_data is None:
        return MissingInput("batch/gae")

    # 1. Stale Policy Detection (On-Policy requirement)
    if node.params.get("strict_on_policy", False):
        param_store_handle = node.params.get("param_store_handle")
        param_store = (
            getattr(context, "param_stores", {}).get(param_store_handle)
            if param_store_handle
            else None
        )

        data_version = batch.get("policy_version") if isinstance(batch, dict) else None
        if param_store and data_version is not None:
            if data_version != param_store.version:
                raise ValueError(
                    f"STALE POLICY DETECTED: Data version {data_version} != current version {param_store.version}"
                )

    model_handle = node.params.get("model_handle", "ppo_net")
    ac_net = context.get_model(model_handle)
    clip_epsilon = node.params["clip_epsilon"]
    entropy_coef = node.params.get("entropy_coef", 0.01)
    critic_coef = node.params.get("critic_coef", 0.5)

    obs = batch["obs"]
    actions = batch["action"].long()
    old_log_probs = batch["log_prob"]
    old_values = batch.get("value")
    advantages = gae_data["advantages"]
    returns = gae_data["returns"]

    # Normalize advantages
    if node.params.get("normalize_advantages", True):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    probs, values = ac_net(obs)
    values = values.view(-1)
    dist = torch.distributions.Categorical(probs)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    # 1. Policy Loss
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    # 2. Value Loss (with optional clipping)
    if node.params.get("clip_value_loss", True) and old_values is not None:
        v_clipped = old_values + torch.clamp(
            values - old_values, -clip_epsilon, clip_epsilon
        )
        v_loss_unclipped = (values - returns) ** 2
        v_loss_clipped = (v_clipped - returns) ** 2
        critic_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
    else:
        critic_loss = 0.5 * nn.functional.mse_loss(values, returns)

    # 3. Combined Loss
    loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy

    # 4. Additional Metrics
    with torch.no_grad():
        log_ratio = new_log_probs - old_log_probs
        approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
        clip_fraction = (torch.abs(ratio - 1.0) > clip_epsilon).float().mean()

        # Explained Variance
        y_pred, y_true = values, returns
        var_y = torch.var(y_true)
        explained_var = (
            1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8)
            if var_y > 1e-8
            else torch.tensor(0.0)
        )

    return {
        "loss": loss,
        "policy_loss": actor_loss.item(),
        "value_loss": critic_loss.item(),
        "entropy": entropy.item(),
        "approx_kl": approx_kl.item(),
        "clip_fraction": clip_fraction.item(),
        "explained_variance": explained_var.item(),
    }


def op_optimizer_step(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
):
    """
    Optimizer step operator.

    Args:
        node: The graph node.
        inputs: Input values (must contain 'loss').
        context: Execution context for optimizer access.

    Returns:
        The loss item value.
    """
    optimizer_handle = node.params.get("optimizer_handle", "main_opt")
    opt_state = context.get_optimizer(optimizer_handle)

    loss_in = inputs.get("loss")
    if loss_in is None:
        return MissingInput("loss")

    # If loss is a dict (from op_ppo_objective), extract the actual loss tensor
    metrics = {}
    if isinstance(loss_in, dict):
        metrics.update({k: v for k, v in loss_in.items() if k != "loss"})
        loss = loss_in["loss"]
    else:
        loss = loss_in

    # Perform optimization step
    step_results = opt_state.step(loss)
    
    # Merge metrics
    metrics.update(step_results)
    return metrics


def register_ppo_operators():
    """Register all PPO related operators."""
    register_operator("PPO_PolicyActor", op_policy_actor)
    register_operator("PPO_GAE", op_gae)
    register_operator("PPO_Objective", op_ppo_objective)
    register_operator("PPO_Optimizer", op_optimizer_step)
