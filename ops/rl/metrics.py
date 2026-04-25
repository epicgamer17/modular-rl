import torch
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.refs import RuntimeValue
from runtime.registry import OperatorSpec, PortSpec, Scalar, TransitionBatch

METRICS_SINK_SPEC = OperatorSpec.create(
    "MetricsSink",
    inputs={
        "loss": PortSpec(spec=Scalar("float32"), required=False),
        "avg_q": PortSpec(spec=Scalar("float32"), required=False),
        "reward": PortSpec(spec=Scalar("float32"), required=False),
        "epsilon": PortSpec(spec=Scalar("float32"), required=False),
        "replay_size": PortSpec(spec=Scalar("int64"), required=False),
        "batch": PortSpec(spec=TransitionBatch, required=False),
        "default": PortSpec(spec=Scalar("float32"), required=False, variadic=True),
    },
    outputs={},
    pure=False,
    stateful=True,
    allowed_contexts={"actor", "learner"},
    side_effects=["logging"],
    math_category="control",
    differentiable=False,
    creates_grad=False,
    consumes_grad=False,
    updates_params=False,
)


def _is_valid(val: Any) -> bool:
    """Helper to check if a value is real data and not a control-flow RuntimeValue."""
    if isinstance(val, RuntimeValue):
        return val.has_data
    return val is not None


# TODO: what should be handled here and what should be handled in observations/ ?
def op_metrics_sink(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> Dict[str, Any]:
    """
    Collects and logs metrics.
    """
    context = context or ExecutionContext()

    loss = inputs.get("loss")
    avg_q = inputs.get("avg_q")
    reward = inputs.get("reward")
    epsilon = inputs.get("epsilon")
    replay_size = inputs.get("replay_size")

    if replay_size is None:
        buffer_id = node.params.get("buffer_id", "main")
        try:
            rb = context.get_buffer(buffer_id)
            if rb is not None:
                replay_size = len(rb)
        except (KeyError, AttributeError):
            pass

    metrics = {
        "actor_step": context.actor_step,
        "learner_step": context.learner_step,
        "episode": context.episode_count,
        "sync_count": context.sync_step,
    }

    if _is_valid(loss):
        if isinstance(loss, dict):
            for k, v in loss.items():
                val = v.item() if hasattr(v, "item") else v
                metrics[k] = (
                    float(val) if isinstance(val, (int, float, torch.Tensor)) else val
                )
        else:
            metrics["loss"] = (
                loss.mean().item()
                if hasattr(loss, "mean")
                else (float(loss) if isinstance(loss, (float, int)) else loss.item())
            )

    if _is_valid(avg_q):
        metrics["avg_q"] = (
            avg_q.mean().item() if hasattr(avg_q, "mean") else float(avg_q)
        )

    if _is_valid(reward):
        metrics["reward"] = (
            reward.mean().item() if hasattr(reward, "mean") else float(reward)
        )

    if _is_valid(epsilon):
        metrics["epsilon"] = float(epsilon)

    if _is_valid(replay_size):
        metrics["replay_size"] = int(replay_size)

    handled = {"loss", "avg_q", "reward", "epsilon", "replay_size", "batch"}
    for port_name, val in inputs.items():
        if port_name in handled or not _is_valid(val):
            continue
        if isinstance(val, dict):
            for k, v in val.items():
                scalar = v.item() if hasattr(v, "item") else v
                if isinstance(scalar, (int, float)):
                    metrics[k] = float(scalar)
        else:
            scalar = val.item() if hasattr(val, "item") else val
            if isinstance(scalar, (int, float)):
                metrics[port_name] = float(scalar)

    # Use the new event-driven observability module
    from observability.tracing.event_schema import get_emitter

    emitter = get_emitter()

    # Log all metrics as events
    for k, v in metrics.items():
        emitter.emit_metric(name=k, value=v, step=context.actor_step)

    return metrics
