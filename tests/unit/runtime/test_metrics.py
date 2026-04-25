import pytest
import torch
from core.graph import Graph, NODE_TYPE_METRICS_SINK
from runtime.context import ExecutionContext
from runtime.executor import execute, register_operator
from ops.rl.metrics import op_metrics_sink, METRICS_SINK_SPEC
from runtime.registry import register_spec

register_spec(NODE_TYPE_METRICS_SINK, METRICS_SINK_SPEC)
register_operator(NODE_TYPE_METRICS_SINK, op_metrics_sink)

from core.graph import NODE_TYPE_SOURCE

register_operator(NODE_TYPE_SOURCE, lambda n, i, context=None: None)

# Set module level marker as per RULE[testing-standards.md]
pytestmark = pytest.mark.unit


def test_metrics_sink_output_dict():
    """Verifies that MetricsSink returns a dictionary of processed metrics."""
    from core.graph import NODE_TYPE_SOURCE

    graph = Graph()
    graph.add_node("loss_src", NODE_TYPE_SOURCE)
    graph.add_node("reward_src", NODE_TYPE_SOURCE)
    graph.add_node("eps_src", NODE_TYPE_SOURCE)
    graph.add_node("metrics", NODE_TYPE_METRICS_SINK)

    graph.add_edge("loss_src", "metrics", dst_port="loss")
    graph.add_edge("reward_src", "metrics", dst_port="reward")
    graph.add_edge("eps_src", "metrics", dst_port="epsilon")

    ctx = ExecutionContext(learner_step=1, actor_step=10, sync_step=2, episode_count=1)
    inputs = {
        "loss_src": torch.tensor(0.5),
        "reward_src": torch.tensor(10.0),
        "eps_src": 0.1,
    }

    outputs = execute(graph, inputs, context=ctx)
    metrics = outputs["metrics"].data

    assert metrics["loss"] == 0.5
    assert metrics["reward"] == 10.0
    assert metrics["epsilon"] == 0.1
    assert metrics["actor_step"] == 10
    assert metrics["sync_count"] == 2
    assert "sps" in metrics
