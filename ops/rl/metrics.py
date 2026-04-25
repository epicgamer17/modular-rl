import torch
from typing import Dict, Any, Optional
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.refs import RuntimeValue
from runtime.services.metrics import MetricsStore

def _is_valid(val: Any) -> bool:
    """Helper to check if a value is real data and not a control-flow RuntimeValue."""
    if isinstance(val, RuntimeValue):
        return val.has_data
    return val is not None

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
                metrics[k] = float(val) if isinstance(val, (int, float, torch.Tensor)) else val
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

    rates = MetricsStore.default().update(context.actor_step, context.learner_step)
    metrics["sps"] = rates["sps"]
    metrics["learner_ups"] = rates["ups"]

    log_frequency = node.params.get("log_frequency", 100)
    if context.learner_step % log_frequency == 0:
        loss_str = f"{metrics['loss']:.4f}" if "loss" in metrics else "N/A"
        q_str = f"{metrics['avg_q']:.4f}" if "avg_q" in metrics else "N/A"
        rew_str = f"{metrics['reward']:.2f}" if "reward" in metrics else "N/A"
        eps_str = f"{metrics['epsilon']:.3f}" if "epsilon" in metrics else "N/A"

        print(
            f"[Metrics] Step {context.actor_step} | Learner {context.learner_step} | "
            f"Loss: {loss_str} | AvgQ: {q_str} | Reward: {rew_str} | Eps: {eps_str} | "
            f"SPS: {metrics['sps']:.1f} | UPS: {metrics['learner_ups']:.1f} | Replay: {metrics.get('replay_size', 0)}"
        )

    return metrics
