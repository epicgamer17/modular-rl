import torch
import torch.nn as nn
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint
from typing import Dict, Any, Optional, List

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
        # obs is [B, ...]
        probs, value = functional_call(ac_net, params, (obs,))
        dist = torch.distributions.Categorical(logits=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

    return {
        "action": action,
        "log_prob": log_prob,
        "values": value.squeeze(-1),
        "policy_version": version,
    }


from runtime.operators.advantage import (
    op_advantage_estimation,
    op_gae,
    op_td_lambda,
    op_mc
)

# Aliases for backward compatibility in tests
op_ppo_gae = op_gae

def op_ppo_objective(*args, **kwargs):
    raise ImportError(
        "op_ppo_objective has been removed. Use decomposed primitives (SurrogateLoss, ValueLoss, Entropy) instead."
    )





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
    
    # Increment learner step counter in context
    if context:
        context.learner_step += 1

    # Merge metrics
    metrics.update(step_results)
    return metrics


def op_policy_ratio(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """PolicyRatio(new_log_prob, old_log_prob) -> exp(new - old)"""
    new_log_prob = inputs.get("new_log_prob")
    old_log_prob = inputs.get("old_log_prob")
    if new_log_prob is None:
        return MissingInput("new_log_prob")
    if old_log_prob is None:
        return MissingInput("old_log_prob")
    return torch.exp(new_log_prob - old_log_prob)


def op_ppo_clip(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """Clip(x, eps) -> clamp(x, 1-eps, 1+eps)"""
    x = inputs.get("x")
    eps = node.params.get("eps", 0.2)
    if x is None:
        return MissingInput("x")
    return torch.clamp(x, 1.0 - eps, 1.0 + eps)


def op_surrogate_loss(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """SurrogateLoss(ratio, advantages, clipped_ratio) -> -min(ratio*A, clipped*A).mean()"""
    ratio = inputs.get("ratio")
    clipped_ratio = inputs.get("clipped_ratio")
    advantages = inputs.get("advantages")
    if ratio is None:
        return MissingInput("ratio")
    if clipped_ratio is None:
        return MissingInput("clipped_ratio")
    if advantages is None:
        return MissingInput("advantages")

    # Enforce no-broadcast policy
    assert (
        ratio.shape == advantages.shape
    ), f"Ratio shape {ratio.shape} must match Advantages shape {advantages.shape}"
    assert (
        clipped_ratio.shape == advantages.shape
    ), f"Clipped Ratio shape {clipped_ratio.shape} must match Advantages shape {advantages.shape}"

    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    return -torch.min(surr1, surr2).mean()


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


def op_entropy(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """Entropy(logits) -> entropy(dist).mean()"""
    logits = inputs.get("logits")
    if logits is None:
        return MissingInput("logits")
    dist = torch.distributions.Categorical(logits=logits)
    return dist.entropy().mean()


def op_weighted_sum(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """WeightedSum(**inputs) -> sum(weight * tensor)"""
    tensors = []
    for name, tensor in inputs.items():
        if isinstance(tensor, (torch.Tensor, float, int)):
            weight = node.params.get(name, 1.0)
            tensors.append(weight * tensor)
    
    if not tensors:
        return torch.tensor(0.0)
        
    total = tensors[0]
    for t in tensors[1:]:
        total = total + t
        
    return total


def op_ppo_train_forward(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> Dict[str, torch.Tensor]:
    """PPO_Forward(obs) -> {logits, values}"""
    obs = inputs.get("obs")
    if obs is None:
        return MissingInput("obs")

    model_handle = node.params.get("model_handle", "ppo_net")
    ac_net = context.get_model(model_handle)

    if node.params.get("activation_checkpoint", False):
        logits, values = checkpoint(lambda x: ac_net(x), obs, use_reentrant=False)
    else:
        logits, values = ac_net(obs)

    return {"logits": logits, "values": values.squeeze(-1)}


def op_log_prob(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """LogProb(logits, action) -> log_prob"""
    logits = inputs.get("logits")
    action = inputs.get("action")
    if logits is None or action is None:
        return MissingInput("logits/action")

    dist = torch.distributions.Categorical(logits=logits)
    return dist.log_prob(action.long())


def op_ppo_sample_batch(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> Dict[str, torch.Tensor]:
    """SampleBatch(buffer_id) -> all transitions from buffer"""
    buffer_id = node.params.get("buffer_id", "main")
    buffer = context.get_buffer(buffer_id)
    return buffer.get_all()


def op_loop(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> List[Dict[str, Any]]:
    """Loop(iterations, body_graph) -> execute body_graph N times"""
    from runtime.executor import execute

    iterations = node.params.get("iterations", 1)
    body_graph = node.params.get("body_graph")
    if not body_graph:
        return []

    results = []
    for _ in range(iterations):
        res = execute(body_graph, initial_inputs=inputs, context=context)
        results.append(res)
    return results


def op_ppo_minibatch_iterator(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> List[Dict[str, Any]]:
    """MinibatchIterator(batch, advantages, returns, body_graph) -> loop over minibatches"""
    from runtime.executor import execute
    import numpy as np

    batch = inputs.get("batch")
    advantages = inputs.get("advantages")
    returns = inputs.get("returns")
    minibatch_size = node.params.get("minibatch_size", 64)
    body_graph = node.params.get("body_graph")

    if not batch or not body_graph:
        return []

    # Combine extra data
    extra_data = {"advantages": advantages, "returns": returns}

    # Slice logic
    total_size = batch.obs.shape[0]
    indices = np.arange(total_size)
    np.random.shuffle(indices)

    results = []
    for start in range(0, total_size, minibatch_size):
        end = start + minibatch_size
        mb_indices = indices[start:end]

        from core.batch import TransitionBatch
        
        # Create a new TransitionBatch for the minibatch
        minibatch = TransitionBatch(
            obs=batch.obs[mb_indices],
            action=batch.action[mb_indices],
            reward=batch.reward[mb_indices],
            next_obs=batch.next_obs[mb_indices],
            done=batch.done[mb_indices],
            log_prob=batch.log_prob[mb_indices] if batch.log_prob is not None else None,
            value=batch.value[mb_indices] if batch.value is not None else None,
            terminated=batch.terminated[mb_indices] if batch.terminated is not None else None,
            truncated=batch.truncated[mb_indices] if batch.truncated is not None else None,
            policy_version=batch.policy_version[mb_indices] if batch.policy_version is not None else None,
            advantages=advantages[mb_indices] if advantages is not None else None,
            returns=returns[mb_indices] if returns is not None else None,
        )

        res = execute(body_graph, initial_inputs={"traj_in": minibatch}, context=context)
        results.append(res)
    return results


def register_ppo_operators():
    """Register all PPO related operators."""
    register_operator("PolicyForward", op_policy_actor)
    from runtime.operators.advantage import op_gae, op_td_lambda, op_mc
    register_operator("AdvantageEstimation", op_advantage_estimation)
    register_operator("PPO_GAE", op_advantage_estimation)
    register_operator("GAE", op_gae)
    register_operator("TDLambda", op_td_lambda)
    register_operator("MC", op_mc)
    register_operator("PPO_Optimizer", op_optimizer_step)

    from runtime.specs import OperatorSpec, register_spec, PortSpec
    from core.schema import TensorSpec, TransitionBatchSpec
    
    # 1. Value Loss Spec
    val_loss_spec = OperatorSpec.create(
        name="ValueLoss",
        inputs={
            "values": TensorSpec(shape=(-1,), dtype="float32"),
            "returns": TensorSpec(shape=(-1,), dtype="float32"),
            "old_values": PortSpec(spec=TensorSpec(shape=(-1,), dtype="float32"), required=False),
        },
        outputs={"loss": TensorSpec(shape=(), dtype="float32")},
        math_category="loss"
    )
    
    # 2. Surrogate Loss Spec
    surr_loss_spec = OperatorSpec.create(
        name="SurrogateLoss",
        inputs={
            "ratio": TensorSpec(shape=(-1,), dtype="float32"),
            "clipped_ratio": TensorSpec(shape=(-1,), dtype="float32"),
            "advantages": TensorSpec(shape=(-1,), dtype="float32"),
        },
        outputs={"loss": TensorSpec(shape=(), dtype="float32")},
        domain_tags={"policy_gradient"},
        math_category="loss"
    )
    
    # 3. Entropy Spec
    entropy_spec = OperatorSpec.create(
        name="Entropy",
        inputs={
            "logits": TensorSpec(shape=(-1, -1), dtype="float32"),
        },
        outputs={"entropy": TensorSpec(shape=(), dtype="float32")},
        domain_tags={"policy_gradient"},
        math_category="reduction"
    )

    # 4. Advantage Estimation Spec
    adv_spec = OperatorSpec.create(
        name="AdvantageEstimation",
        inputs={
            "batch": TransitionBatchSpec,
            "next_value": TensorSpec(shape=(-1,), dtype="float32"),
            "next_terminated": TensorSpec(shape=(-1,), dtype="bool"),
        },
        outputs={
            "advantages": TensorSpec(shape=(-1,), dtype="float32"),
            "returns": TensorSpec(shape=(-1,), dtype="float32"),
        },
        domain_tags={"policy_gradient"},
        math_category="elementwise"
    )

    # 5. Optimizer Spec
    opt_spec = OperatorSpec.create(
        name="PPO_Optimizer",
        inputs={"loss": TensorSpec(shape=(), dtype="float32")},
        outputs={},
        pure=False,
        stateful=True,
        updates_params=True,
        math_category="optimizer"
    )
    
    # 6. Sample Batch Spec
    sample_spec = OperatorSpec.create(
        name="SampleBatch",
        inputs={},
        outputs={"batch": TransitionBatchSpec},
        pure=False,
        stateful=True,
        reads_buffer=True,
        math_category="buffer_io"
    )

    # 7. LogProb Spec
    log_prob_spec = OperatorSpec.create(
        name="LogProb",
        inputs={
            "logits": TensorSpec(shape=(-1, -1), dtype="float32"), # [B, A]
            "action": TensorSpec(shape=(-1,), dtype="int64"),      # [B]
        },
        outputs={"log_prob": TensorSpec(shape=(-1,), dtype="float32")},
        domain_tags={"policy_gradient"},
        math_category="distribution"
    )
    
    # 8. PolicyRatio Spec
    ratio_spec = OperatorSpec.create(
        name="PolicyRatio",
        inputs={
            "new_log_prob": TensorSpec(shape=(-1,), dtype="float32"),
            "old_log_prob": TensorSpec(shape=(-1,), dtype="float32"),
        },
        outputs={"ratio": TensorSpec(shape=(-1,), dtype="float32")},
        domain_tags={"policy_gradient"},
        math_category="elementwise"
    )

    # 9. Clip Spec
    clip_spec = OperatorSpec.create(
        name="Clip",
        inputs={"x": TensorSpec(shape=(-1,), dtype="float32")},
        outputs={"clipped_x": TensorSpec(shape=(-1,), dtype="float32")},
        domain_tags={"policy_gradient"},
        math_category="elementwise"
    )
    
    # 10. WeightedSum Spec
    weighted_sum_spec = OperatorSpec.create(
        name="WeightedSum",
        inputs={
            "default": PortSpec(spec=TensorSpec(shape=(), dtype="float32"), required=False, variadic=True),
        },
        outputs={"sum": TensorSpec(shape=(), dtype="float32")},
        math_category="elementwise"
    )

    register_operator("PPO_Forward", op_ppo_train_forward)
    register_operator("LogProb", op_log_prob, spec=log_prob_spec)
    register_operator("PolicyRatio", op_policy_ratio, spec=ratio_spec)
    register_operator("Clip", op_ppo_clip, spec=clip_spec)
    register_operator("SurrogateLoss", op_surrogate_loss, spec=surr_loss_spec)
    register_operator("ValueLoss", op_value_loss, spec=val_loss_spec)
    register_operator("Entropy", op_entropy, spec=entropy_spec)
    register_operator("WeightedSum", op_weighted_sum, spec=weighted_sum_spec)
    register_operator("AdvantageEstimation", op_advantage_estimation, spec=adv_spec)
    register_operator("PPO_Optimizer", op_optimizer_step, spec=opt_spec)

    # Loop Primitives
    register_operator("SampleBatch", op_ppo_sample_batch, spec=sample_spec)
    register_operator("Loop", op_loop)
    register_operator("MinibatchIterator", op_ppo_minibatch_iterator)
