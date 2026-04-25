import torch
from torch.func import functional_call
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.signals import MissingInput
from runtime.registry import OperatorSpec, TensorSpec, Scalar, PortSpec

POLICY_RATIO_SPEC = OperatorSpec.create(
    name="PolicyRatio",
    inputs={
        "new_log_prob": TensorSpec(shape=(-1,), dtype="float32"),
        "old_log_prob": TensorSpec(shape=(-1,), dtype="float32"),
    },
    outputs={"ratio": TensorSpec(shape=(-1,), dtype="float32")},
    domain_tags={"policy_gradient"},
    math_category="elementwise",
    allowed_contexts={"actor", "learner"},
    differentiable=True,
    creates_grad=True,
    consumes_grad=False,
    updates_params=False,
)

GREEDY_ACTION_SPEC = OperatorSpec.create(
    name="GreedyAction",
    inputs={"input": PortSpec(spec=None)},
    outputs={"output": Scalar("int64")},
    pure=True,
    math_category="distribution",
    allowed_contexts={"actor", "learner"},
    differentiable=False,
    creates_grad=False,
    consumes_grad=False,
    updates_params=False,
)


def op_policy_actor(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
):
    """
    Operator for PPO policy actor.
    Expects input 'obs'.
    Returns dictionary with action, log_prob, values, and policy_version.
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


def op_policy_forward(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> Dict[str, torch.Tensor]:
    """PPO Forward pass for training: obs -> {logits, values}"""
    obs = inputs.get("obs")
    if obs is None:
        return MissingInput("obs")

    model_handle = node.params.get("model_handle", "ppo_net")
    ac_net = context.get_model(model_handle)

    # Stale policy detection
    # TODO: clean this up its messy but works for now
    version = inputs.get("policy_version")
    if version is not None:
        current_version = context.policy_versions.get(model_handle, 0)
        # Handle both tensor and scalar versions
        if hasattr(version, "numel") and version.numel() == 1:
            # It's a scalar tensor
            version = version.item()
        elif hasattr(version, "numel") and version.numel() > 1:
            # It's a batch tensor - check if any are stale
            if torch.any(version < current_version):
                raise RuntimeError(f"Stale Policy Error: Batch has stale policies")
        elif version < current_version:
            raise RuntimeError(
                f"Stale Policy Error: Batch version {version} < current {current_version}"
            )

    # We don't use functional_call here because we want gradients
    logits, values = ac_net(obs)
    return {"logits": logits, "values": values.squeeze(-1)}


def op_policy_ratio(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """Computes ratio = exp(new_log_prob - old_log_prob)."""
    new_log_prob = inputs.get("new_log_prob")
    old_log_prob = inputs.get("old_log_prob")
    if new_log_prob is None or old_log_prob is None:
        return MissingInput("log_probs")
    return torch.exp(new_log_prob - old_log_prob)


def op_greedy_action(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> torch.Tensor:
    """Argmax over logits/probs: [B, A] -> [B]."""
    x = inputs.get("input")
    if x is None:
        return MissingInput("input")
    return torch.argmax(x, dim=-1)
