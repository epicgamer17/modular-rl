import pytest
import torch
from core.graph import Graph, NODE_TYPE_REPLAY_QUERY, NODE_TYPE_SOURCE
from runtime.context import ExecutionContext
from runtime.state import ReplayBuffer
from runtime.runtime import LearnerRuntime
from runtime.executor import register_operator

# Set module level marker as per RULE[testing-standards.md]
pytestmark = pytest.mark.unit

def test_declarative_replay_query_and_min_size():
    """
    Test 7A: Verify that ReplayQuery node handles min_size and collation,
    allowing a standard LearnerRuntime to run without subclassing.
    """
    rb = ReplayBuffer(capacity=100)
    
    # Mock collator that just adds a flag
    def mock_collator(batch):
        return {"data": batch, "collated": True}
        
    train_graph = Graph()
    train_graph.add_node("traj_in", NODE_TYPE_SOURCE)
    train_graph.add_node("sampler", NODE_TYPE_REPLAY_QUERY, params={
        "replay_buffer": rb,
        "batch_size": 2,
        "min_size": 5,
        "collator": mock_collator
    })
    # sampler should not have an edge from traj_in unless it actually needs it
    
    # Track if the downstream node was called with valid data
    received_batch = None
    def op_process(node, inputs, context=None):
        nonlocal received_batch
        received_batch = list(inputs.values())[0]
        return "ok"
        
    register_operator("Process", op_process)
    train_graph.add_node("proc", "Process")
    train_graph.add_edge("sampler", "proc")
    
    learner = LearnerRuntime(train_graph, replay_buffer=rb)
    ctx = ExecutionContext()
    
    # 1. Size 0 < 5: sampler should return None, proc should receive None
    learner.update_step(context=ctx)
    assert received_batch is None
    assert ctx.learner_step == 1
    
    # 2. Add some data but still < 5
    for i in range(3):
        rb.add({"val": i})
    learner.update_step(context=ctx)
    assert received_batch is None
    assert ctx.learner_step == 2
    
    # 3. Add more data to reach min_size=5
    for i in range(2):
        rb.add({"val": i + 3})
    learner.update_step(context=ctx)
    
    assert received_batch is not None
    assert received_batch["collated"] is True
    assert len(received_batch["data"]) == 2
    assert ctx.learner_step == 3

def test_learner_runtime_no_override_dqn_style():
    """Verify standard LearnerRuntime can execute a full DQN-like graph."""
    # This is essentially a simplified version of dqn.py without the actual NN
    rb = ReplayBuffer(capacity=100)
    for i in range(10): rb.add({"x": i})
    
    graph = Graph()
    graph.add_node("sampler", NODE_TYPE_REPLAY_QUERY, params={
        "replay_buffer": rb,
        "batch_size": 4,
        "min_size": 5
    })
    
    from runtime.values import NoOp
    def op_loss(node, inputs, context=None):
        batch = list(inputs.values())[0]
        if isinstance(batch, NoOp) or batch is None:
            return NoOp()
        return torch.tensor(1.0, requires_grad=True)
        
    def op_opt(node, inputs, context=None):
        loss = list(inputs.values())[0]
        if isinstance(loss, NoOp) or loss is None:
            return NoOp()
        return "stepped"
        
    register_operator("LossNode", op_loss)
    register_operator("OptNode", op_opt)
    
    graph.add_node("loss", "LossNode")
    graph.add_node("opt", "OptNode")
    graph.add_edge("sampler", "loss")
    graph.add_edge("loss", "opt")
    
    learner = LearnerRuntime(graph, replay_buffer=rb)
    results = learner.update_step()
    assert results["opt"] == "stepped"
    
    # Clear buffer to trigger min_size
    rb.buffer.clear()
    results = learner.update_step()
    # Now it should be a MissingInput/Skipped because buffer is empty
    from runtime.values import RuntimeValue
    assert isinstance(results["opt"], RuntimeValue)
    assert not results["opt"].has_data
