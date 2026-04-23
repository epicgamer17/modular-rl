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
    """Internal helper to track throughput (SPS) and other rolling metrics."""

    def __init__(self):
        self.start_time = time.time()
        self.last_actor_step = 0
        self.last_time = self.start_time

    def get_sps(self, current_actor_step: int) -> float:
        now = time.time()
        elapsed = now - self.last_time
        if elapsed < 0.001:
            return 0.0

        steps = current_actor_step - self.last_actor_step
        sps = steps / elapsed

        # Update markers for next call
        self.last_time = now
        self.last_actor_step = current_actor_step
        return sps


# Global tracker instance
_TRACKER = MetricsTracker()


def op_metrics_sink(
    node: Node, inputs: Dict[str, Any], context: Optional[ExecutionContext] = None
) -> Dict[str, Any]:
    """
    Collects and logs metrics.

    Inputs can contain metric values directly (if key matches) or within dictionaries.
    Keys tracked: loss, q_values, reward, epsilon, replay_size.
    """
    context = context or ExecutionContext()

    # 1. Extract values from named ports
    loss = inputs.get("loss")
    avg_q = inputs.get("avg_q")
    reward = inputs.get("reward")
    epsilon = inputs.get("epsilon")
    batch = inputs.get("batch")

    # TODO: Clean this up, fallbacks are messy
    # Fallback: if not on explicit ports, look inside batch dict
    if isinstance(batch, dict):
        loss = loss if loss is not None else batch.get("loss")
        avg_q = avg_q if avg_q is not None else batch.get("avg_q")
        reward = reward if reward is not None else batch.get("reward")
        epsilon = epsilon if epsilon is not None else batch.get("epsilon")

    # Try to extract replay_size from batch if available
    replay_size = None
    if isinstance(batch, dict) and "replay_size" in batch:
        replay_size = batch["replay_size"]

    # If not in inputs, check params for replay_buffer
    if replay_size is None:
        rb = node.params.get("replay_buffer")
        if rb is None and context:
            buffer_id = node.params.get("buffer_id", "main")
            try:
                rb = context.get_buffer(buffer_id)
            except (KeyError, AttributeError):
                pass

        if rb is not None:
            replay_size = len(rb)

    metrics = {
        "actor_step": context.actor_step,
        "learner_step": context.learner_step,
        "episode": context.episode_count,
        "sync_count": context.sync_step,
    }

    if _is_valid(loss):
        metrics["loss"] = float(loss) if isinstance(loss, (float, int)) else loss.item()

    if _is_valid(avg_q) and hasattr(avg_q, "mean"):
        metrics["avg_q"] = avg_q.mean().item()

    if _is_valid(reward):
        if hasattr(reward, "mean"):
            metrics["reward"] = reward.mean().item()
        else:
            metrics["reward"] = float(reward)

    if _is_valid(epsilon):
        metrics["epsilon"] = float(epsilon)

    if _is_valid(replay_size):
        metrics["replay_size"] = int(replay_size)

    # Throughput (SPS)
    metrics["sps"] = _TRACKER.get_sps(context.actor_step)

    # Log every N learner steps if configured
    log_frequency = node.params.get("log_frequency", 100)
    if context.learner_step % log_frequency == 0:
        loss_str = f"{metrics['loss']:.4f}" if "loss" in metrics else "N/A"
        q_str = f"{metrics['avg_q']:.4f}" if "avg_q" in metrics else "N/A"
        print(
            f"[Metrics] Step {context.actor_step} | Learner {context.learner_step} | "
            f"Loss: {loss_str} | AvgQ: {q_str} | "
            f"SPS: {metrics['sps']:.1f} | Replay: {metrics.get('replay_size', 0)}"
        )

    return metrics
