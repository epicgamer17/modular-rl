"""
Operator for metrics collection and logging.
"""

from typing import Dict, Any, Optional, List
import time
import torch
from core.graph import Node
from runtime.context import ExecutionContext
from runtime.values import RuntimeValue


def _is_valid(val: Any) -> bool:
    """Helper to check if a value is real data and not a control-flow RuntimeValue."""
    if isinstance(val, RuntimeValue):
        return val.has_data
    return val is not None


class MetricsTracker:
    """Internal helper to track throughput (SPS/UPS) and other rolling metrics."""

    def __init__(self):
        self.start_time = time.time()
        self.last_actor_step = 0
        self.last_learner_step = 0
        self.last_time = self.start_time

    def update(
        self, current_actor_step: int, current_learner_step: int
    ) -> Dict[str, float]:
        now = time.time()
        elapsed = now - self.last_time
        if elapsed < 0.001:
            return {"sps": 0.0, "ups": 0.0}

        actor_delta = current_actor_step - self.last_actor_step
        learner_delta = current_learner_step - self.last_learner_step

        sps = actor_delta / elapsed
        ups = learner_delta / elapsed

        # Update markers for next call
        self.last_time = now
        self.last_actor_step = current_actor_step
        self.last_learner_step = current_learner_step

        return {"sps": sps, "ups": ups}


# Global tracker instance
_TRACKER = MetricsTracker()


def op_metrics_sink(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> Dict[str, Any]:
    """
    Collects and logs metrics.

    Inputs can contain metric values directly (if key matches) or within dictionaries.
    Keys tracked: loss, q_values, reward, epsilon, replay_size, learner_ups.
    """
    context = context or ExecutionContext()

    # 1. Extract values from named ports
    loss = inputs.get("loss")
    avg_q = inputs.get("avg_q")
    reward = inputs.get("reward")
    epsilon = inputs.get("epsilon")
    replay_size = inputs.get("replay_size")

    # If not in inputs, check params for replay_buffer
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

    # Throughput (SPS and UPS)
    rates = _TRACKER.update(context.actor_step, context.learner_step)
    metrics["sps"] = rates["sps"]
    metrics["learner_ups"] = rates["ups"]

    # Log every N learner steps if configured
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
