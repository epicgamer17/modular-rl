import pytest
import torch
from core.graph import Graph, NODE_TYPE_METRICS_SINK
from runtime.context import ExecutionContext
from runtime.executor import execute, register_operator
from runtime.operators.metrics import op_metrics_sink

# Set module level marker as per RULE[testing-standards.md]
pytestmark = pytest.mark.unit

def test_metrics_sink_emits_periodically(capsys):
    """
    Test 8A: Metrics emitted every N learner steps.
    Verifies that MetricsSink operator logs to stdout at the specified frequency.
    """
    from core.graph import NODE_TYPE_SOURCE
    graph = Graph()
    graph.add_node("data_source", NODE_TYPE_SOURCE)
    graph.add_node("metrics", NODE_TYPE_METRICS_SINK, params={"log_frequency": 5})
    graph.add_edge("data_source", "metrics", dst_port="batch")
    
    # Run multiple times with different learner steps
    for i in range(11):
        ctx = ExecutionContext(learner_step=i, actor_step=i*10)
        # Pass dummy inputs via the source node
        data = {
            "loss": torch.tensor(1.0 - i/10),
            "q_values": torch.ones(5) * i,
            "replay_size": 100 + i
        }
        execute(graph, {"data_source": data}, context=ctx)
        
    captured = capsys.readouterr()
    # Should log at step 0, 5, 10
    assert captured.out.count("[Metrics]") == 3
    assert "Step 0" in captured.out
    assert "Step 50" in captured.out
    assert "Step 100" in captured.out
    assert "Replay: 110" in captured.out

def test_metrics_sink_output_dict():
    """Verifies that MetricsSink returns a dictionary of processed metrics."""
    from core.graph import NODE_TYPE_SOURCE
    graph = Graph()
    graph.add_node("data_source", NODE_TYPE_SOURCE)
    graph.add_node("metrics", NODE_TYPE_METRICS_SINK)
    graph.add_edge("data_source", "metrics", dst_port="batch")
    
    ctx = ExecutionContext(learner_step=1, actor_step=10, sync_step=2, episode_count=1)
    data = {
        "loss": torch.tensor(0.5),
        "reward": 10.0,
        "epsilon": 0.1
    }
    
    outputs = execute(graph, {"data_source": data}, context=ctx)
    metrics = outputs["metrics"].data
    
    assert metrics["loss"] == 0.5
    assert metrics["reward"] == 10.0
    assert metrics["epsilon"] == 0.1
    assert metrics["actor_step"] == 10
    assert metrics["sync_count"] == 2
    assert "sps" in metrics
