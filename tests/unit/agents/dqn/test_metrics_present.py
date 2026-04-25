import pytest
import torch
from core.graph import Graph
from runtime.executor import execute, register_operator
from runtime.context import ExecutionContext
from runtime.state import ReplayBuffer, BufferRegistry
from ops.rl.metrics import op_metrics_sink

pytestmark = pytest.mark.unit

def test_dqn_metrics_emission():
    """Verify that all required DQN metrics are emitted by the sink."""
    register_operator("MetricsSink", op_metrics_sink)
    
    graph = Graph()
    from core.graph import NODE_TYPE_SOURCE
    graph.add_node("loss_src", NODE_TYPE_SOURCE)
    graph.add_node("q_src", NODE_TYPE_SOURCE)
    graph.add_node("reward_src", NODE_TYPE_SOURCE)
    graph.add_node("eps_src", NODE_TYPE_SOURCE)
    
    # Mock inputs for the metrics sink
    graph.add_node("metrics", "MetricsSink", params={
        "log_frequency": 1,
        "buffer_id": "main"
    })
    
    # Connect source nodes to ports
    graph.add_edge("loss_src", "metrics", dst_port="loss")
    graph.add_edge("q_src", "metrics", dst_port="avg_q")
    graph.add_edge("reward_src", "metrics", dst_port="reward")
    graph.add_edge("eps_src", "metrics", dst_port="epsilon")
    
    # Setup registries
    rb = ReplayBuffer(capacity=100)
    for i in range(10):
        rb.add({"obs": torch.randn(4), "action": 0, "reward": 1.0, "next_obs": torch.randn(4), "done": False})
        
    registry = BufferRegistry()
    registry.register("main", rb)
    
    # Create context with some state
    ctx = ExecutionContext(
        buffer_registry=registry,
        actor_step=100,
        learner_step=50,
        episode_count=5
    )
    
    # Mock initial inputs
    inputs = {
        "loss_src": torch.tensor(0.5),
        "q_src": torch.tensor(3.2),
        "reward_src": torch.tensor(1.0),
        "eps_src": 0.1
    }
    
    # Execute
    results = execute(graph, initial_inputs=inputs, context=ctx)
    metrics_val = results["metrics"]
    # Unwrap Value if necessary
    metrics = metrics_val.data if hasattr(metrics_val, "data") else metrics_val
    
    # Assert all required keys are present
    assert "epsilon" in metrics, f"Missing epsilon in {metrics.keys()}"
    assert "reward" in metrics, "Missing reward"
    assert "loss" in metrics, "Missing loss"
    assert "avg_q" in metrics, "Missing avg_q"
    assert "replay_size" in metrics, "Missing replay_size"
    assert "learner_ups" in metrics, "Missing learner_ups"
    assert "sps" in metrics, "Missing sps"
    
    # Verify values with tolerance
    assert metrics["epsilon"] == pytest.approx(0.1)
    assert metrics["reward"] == pytest.approx(1.0)
    assert metrics["loss"] == pytest.approx(0.5)
    assert metrics["avg_q"] == pytest.approx(3.2)
    assert metrics["replay_size"] == 10
    assert metrics["actor_step"] == 100
    assert metrics["learner_step"] == 50

if __name__ == "__main__":
    test_dqn_metrics_emission()
    print("Metrics Emission Test Passed!")
