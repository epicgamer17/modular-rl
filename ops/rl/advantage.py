import torch
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput
from runtime.kernels.advantage import gae_advantage, td_lambda_advantage, mc_advantage

ADVANTAGE_KERNELS = {
    "gae": gae_advantage,
    "td_lambda": td_lambda_advantage,
    "mc": mc_advantage
}

def op_advantage_estimation(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
):
    """
    Universal Advantage Estimation operator using functional kernels.
    """
    batch = inputs.get("batch")
    next_value = inputs.get("next_value")
    next_terminated = inputs.get("next_terminated")

    if batch is None or next_value is None or next_terminated is None:
        return MissingInput("batch/next_value/next_terminated")

    method = node.params.get("method", "gae")
    gamma = node.params.get("gamma", 0.99)
    
    kernel_fn = ADVANTAGE_KERNELS.get(method)
    if kernel_fn is None:
        raise ValueError(f"Unknown advantage estimation method: {method}")

    # Handle rollout structure [T, N]
    rewards = batch.reward
    values = batch.value
    
    is_flattened = len(rewards.shape) == 1
    if is_flattened:
        num_envs = node.params.get("num_envs", 1)
        batch_size = rewards.shape[0]
        T = batch_size // num_envs
        
        # Reshape for kernels
        from core.batch import TransitionBatch
        temp_batch = TransitionBatch(
            obs=batch.obs,
            action=batch.action,
            reward=batch.reward.view(T, num_envs),
            next_obs=batch.next_obs,
            done=batch.done.view(T, num_envs) if batch.done is not None else None,
            terminated=batch.terminated.view(T, num_envs) if batch.terminated is not None else None,
            value=batch.value.view(T, num_envs)
        )
    else:
        temp_batch = batch

    # Filter node.params to avoid duplicate arguments in kernel call
    kernel_kwargs = {k: v for k, v in node.params.items() if k not in ["gamma", "method", "num_envs"]}
    
    advantages, returns = kernel_fn(
        temp_batch, 
        next_value, 
        next_terminated, 
        gamma, 
        **kernel_kwargs
    )

    if is_flattened:
        advantages = advantages.view(-1)
        returns = returns.view(-1)

    return {"advantages": advantages, "returns": returns}

def op_gae(node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None):
    node.params["method"] = "gae"
    return op_advantage_estimation(node, inputs, context)

def op_td_lambda(node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None):
    node.params["method"] = "td_lambda"
    return op_advantage_estimation(node, inputs, context)

def op_mc(node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None):
    node.params["method"] = "mc"
    return op_advantage_estimation(node, inputs, context)
