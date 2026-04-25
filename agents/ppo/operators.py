import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.signals import MissingInput
from runtime.context import ExecutionContext


def op_ppo_objective(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> Dict[str, torch.Tensor]:
    """
    Monolithic PPO Objective operator.
    Computes actor loss, critic loss, and entropy loss.
    """
    batch = inputs.get("batch")
    gae = inputs.get("gae")
    if batch is None or gae is None:
        return MissingInput("batch/gae")

    # Extract inputs
    obs = batch.get("obs")
    actions = batch.get("action")
    old_log_probs = batch.get("log_prob")
    advantages = gae.get("advantages")
    returns = gae.get("returns")

    if any(x is None for x in [obs, actions, old_log_probs, advantages]):
        return MissingInput("missing_ppo_inputs")

    # Get model
    model_handle = node.params.get("model_handle", "ppo_net")
    ac_net = context.get_model(model_handle)

    # Forward pass
    logits, values = ac_net(obs)
    values = values.squeeze(-1)

    # Policy distribution
    dist = torch.distributions.Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    # Ratio and Clipping
    ratio = torch.exp(new_log_probs - old_log_probs)
    clip_epsilon = node.params.get("clip_epsilon", 0.2)

    # Optional advantage normalization
    if node.params.get("normalize_advantages", True):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    # Critic Loss
    critic_coef = node.params.get("critic_coef", 0.5)
    critic_loss = (
        0.5 * (values - returns).pow(2).mean()
        if returns is not None
        else torch.tensor(0.0)
    )

    # Entropy Loss
    entropy_coef = node.params.get("entropy_coef", 0.01)

    total_loss = actor_loss + critic_coef * critic_loss - entropy_coef * entropy

    return {
        "loss": total_loss,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "entropy": entropy,
    }


def op_optimizer_step(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
):
    """Optimizer step operator with metrics extraction."""
    optimizer_handle = node.params.get("optimizer_handle", "main_opt")
    opt_state = context.get_optimizer(optimizer_handle)

    loss_in = inputs.get("loss")
    if loss_in is None:
        return MissingInput("loss")

    metrics = {}
    if isinstance(loss_in, dict):
        metrics.update({k: v for k, v in loss_in.items() if k != "loss"})
        loss = loss_in["loss"]
    else:
        loss = loss_in

    step_results = opt_state.step(loss)
    if context:
        context.learner_step += 1

    metrics.update(step_results)
    return metrics


def register_ppo_operators():
    """
    Register all PPO related operators and metadata specs.
    Note: Most operators are registered globally in runtime/executor.py;
    specs are owned by agents/ppo/specs.py.
    """
    from runtime.operator_registry import register_operator
    from agents.ppo.specs import register_ppo_specs

    register_operator("PPO_Objective", op_ppo_objective)
    register_operator("PPO_Optimizer", op_optimizer_step)
    register_ppo_specs()
