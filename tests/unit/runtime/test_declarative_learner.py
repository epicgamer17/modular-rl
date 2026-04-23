import pytest
import torch
from core.graph import Graph, NODE_TYPE_REPLAY_QUERY, NODE_TYPE_SOURCE
from runtime.context import ExecutionContext
from runtime.state import ReplayBuffer, BufferRegistry
from runtime.runtime import LearnerRuntime
from runtime.executor import register_operator
from core.schema import Schema, Field, TensorSpec

# Set module level marker as per RULE[testing-standards.md]
pytestmark = pytest.mark.unit

def test_declarative_replay_query_and_min_size():
    """
    Test 7A: Verify that ReplayQuery node handles min_size and collation,
    allowing a standard LearnerRuntime to run without subclassing.
    """
    rb = ReplayBuffer(capacity=100)
    
    # We use a real schema to trigger the automatic collator creation
    # instead of a mock_collator (which violates P004)
    schema = Schema([Field("val", TensorSpec(shape=(), dtype="int64"))])
        
    train_graph = Graph()
    train_graph.add_node("traj_in", NODE_TYPE_SOURCE)
    train_graph.add_node("sampler", NODE_TYPE_REPLAY_QUERY, 
        params={
            "buffer_id": "main",
            "batch_size": 2,
            "min_size": 5
        },
        schema_out=schema
    )
    
    # Track if the downstream node was called with valid data
    received_batch = None
    def op_process(node, inputs, context=None):
        nonlocal received_batch
        received_batch = inputs["batch"]
        return "ok"
        
    register_operator("Process", op_process)
    train_graph.add_node("proc", "Process")
    train_graph.add_edge("sampler", "proc", dst_port="batch")
    
    registry = BufferRegistry()
    registry.register("main", rb)
    ctx = ExecutionContext(buffer_registry=registry)
    
    learner = LearnerRuntime(train_graph, replay_buffer=rb)
    
    # 1. Size 0 < 5: sampler should return Skipped, proc should receive Skipped
    results = learner.update_step(context=ctx)
    from runtime.values import RuntimeValue
    assert isinstance(results["sampler"], RuntimeValue)
    assert not results["sampler"].has_data
    assert isinstance(results["proc"], RuntimeValue)
    assert not results["proc"].has_data
    assert ctx.learner_step == 1
    
    # 2. Add some data but still < 5
    for i in range(3):
        rb.add({"val": i})
    results = learner.update_step(context=ctx)
    assert not results["sampler"].has_data
    assert ctx.learner_step == 2
    
    # 3. Add more data to reach min_size=5
    for i in range(2):
        rb.add({"val": i + 3})
    results = learner.update_step(context=ctx)
    
    # sampler result is unwrapped by LearnerRuntime into a dict if it has data
    sampler_res = results["sampler"]
    assert isinstance(sampler_res, dict)
    assert len(sampler_res["val"]) == 2
    assert ctx.learner_step == 3

def test_learner_runtime_no_override_dqn_style():
    """Verify standard LearnerRuntime can execute a full DQN-like graph."""
    rb = ReplayBuffer(capacity=100)
    for i in range(10): rb.add({"x": i})
    
    graph = Graph()
    graph.add_node("sampler", NODE_TYPE_REPLAY_QUERY, params={
        "buffer_id": "main",
        "batch_size": 4,
        "min_size": 5
    })
    
    from runtime.values import NoOp, RuntimeValue
    def op_loss(node, inputs, context=None):
        batch = inputs["batch"]
        if isinstance(batch, (NoOp, RuntimeValue)) and not getattr(batch, "has_data", True):
            return NoOp()
        return torch.tensor(1.0, requires_grad=True)
        
    def op_opt(node, inputs, context=None):
        loss = inputs["loss"]
        if isinstance(loss, (NoOp, RuntimeValue)) and not getattr(loss, "has_data", True):
            return NoOp()
        return "stepped"
        
    register_operator("LossNode", op_loss)
    register_operator("OptNode", op_opt)
    
    graph.add_node("loss", "LossNode")
    graph.add_node("opt", "OptNode")
    graph.add_edge("sampler", "loss", dst_port="batch")
    graph.add_edge("loss", "opt", dst_port="loss")
    
    registry = BufferRegistry()
    registry.register("main", rb)
    ctx = ExecutionContext(buffer_registry=registry)
    
    learner = LearnerRuntime(graph, replay_buffer=rb)
    results = learner.update_step(context=ctx)
    assert results["opt"] == "stepped"
    
    # Clear buffer to trigger min_size
    rb.buffer.clear()
    results = learner.update_step(context=ctx)
    # Now it should be a MissingInput/Skipped because buffer is empty
    assert isinstance(results["opt"], RuntimeValue)
    assert not results["opt"].has_data
