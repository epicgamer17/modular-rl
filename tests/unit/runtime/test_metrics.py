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
    graph.add_node("loss_src", NODE_TYPE_SOURCE)
    graph.add_node("q_src", NODE_TYPE_SOURCE)
    graph.add_node("replay_src", NODE_TYPE_SOURCE)
    graph.add_node("metrics", NODE_TYPE_METRICS_SINK, params={"log_frequency": 5})
    
    graph.add_edge("loss_src", "metrics", dst_port="loss")
    graph.add_edge("q_src", "metrics", dst_port="avg_q")
    graph.add_edge("replay_src", "metrics", dst_port="replay_size")
    
    # Run multiple times with different learner steps
    for i in range(11):
        ctx = ExecutionContext(learner_step=i, actor_step=i*10)
        inputs = {
            "loss_src": torch.tensor(1.0 - i/10),
            "q_src": torch.ones(5) * i,
            "replay_src": 100 + i
        }
        execute(graph, inputs, context=ctx)
        
    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")
    
    # Should log at step 0, 5, 10
    metrics_lines = [l for l in lines if "[Metrics]" in l]
    assert len(metrics_lines) == 3
    
    assert "Step 0" in metrics_lines[0]
    assert "Step 50" in metrics_lines[1]
    assert "Step 100" in metrics_lines[2]
    
    assert "Replay: 100" in metrics_lines[0]
    assert "Replay: 105" in metrics_lines[1]
    assert "Replay: 110" in metrics_lines[2]

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
        "eps_src": 0.1
    }
    
    outputs = execute(graph, inputs, context=ctx)
    metrics = outputs["metrics"].data
    
    assert metrics["loss"] == 0.5
    assert metrics["reward"] == 10.0
    assert metrics["epsilon"] == 0.1
    assert metrics["actor_step"] == 10
    assert metrics["sync_count"] == 2
    assert "sps" in metrics
