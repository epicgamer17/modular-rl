import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from core.graph import Graph, NodeId
from agents.dqn.specs import register_dqn_specs
from agents.dqn.operators import register_dqn_operators
from runtime.bootstrap import bootstrap_runtime
from runtime.registry import clear_registry, register_spec, OperatorSpec
from runtime.context import ExecutionContext
from runtime.state import OptimizerState
from runtime.operator_registry import register_operator
from runtime.executor import execute
from runtime.refs import Value
from compiler.pipeline import compile_graph

pytestmark = pytest.mark.unit

@pytest.fixture(autouse=True)
def setup_specs():
    clear_registry()
    bootstrap_runtime()
    register_dqn_specs()
    register_dqn_operators()

def test_backward_updates_grad_buffer():
    """Verifies that the Backward node actually updates gradients in the GradBuffer."""
    g = Graph()
    
    # 1. Forward (Linear model)
    # We'll use a simple Linear layer to test gradients
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 1)
        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    
    # Register a simple Forward op for this test
    def op_forward(node, inputs, context):
        m = context.get_model(node.params["model_handle"])
        return m(inputs["obs"])
    
    register_operator("TestForward", op_forward)
    register_spec("TestForward", OperatorSpec.create(
        name="TestForward",
        inputs={"obs": None},
        outputs={"pred": None},
        differentiable=True,
        creates_grad=True,
        consumes_grad=False,
        updates_params=False,
        parameter_handles=["model_handle"]
    ))

    def op_test_optimizer(node, inputs, context):
        opt_state = context.get_optimizer(node.params["optimizer_handle"])
        opt_state.step()
        return True

    register_operator("TestOptimizer", op_test_optimizer)
    register_spec("TestOptimizer", OperatorSpec.create(
        name="TestOptimizer",
        inputs={"loss": None},
        outputs={"done": None},
        differentiable=False,
        creates_grad=False,
        consumes_grad=True,
        updates_params=True,
        parameter_handles=["model_handle", "optimizer_handle"]
    ))

    g.add_node("obs", "Source")
    g.add_node("forward", "TestForward", params={"model_handle": "m1"})
    g.add_edge("obs", "forward", dst_port="obs")
    
    g.add_node("target", "Source")
    g.add_node("mse", "MSELoss")
    g.add_edge("forward", "mse", src_port="default", dst_port="pred")
    g.add_edge("target", "mse", dst_port="target")
    
    g.add_node("opt", "TestOptimizer", params={"model_handle": "m1", "optimizer_handle": "main_opt"})
    g.add_edge("mse", "opt", dst_port="loss")
    
    # Compile
    compiled = compile_graph(g, context="learner", model_handles={"m1"})
    
    # Setup context
    ctx = ExecutionContext()
    ctx.model_registry.register("m1", model)
    ctx.optimizer_registry.register(
        "main_opt",
        OptimizerState(optim.SGD(model.parameters(), lr=0.01)),
    )
    
    # Execute
    obs = torch.randn(1, 4)
    target = torch.randn(1, 1)
    
    initial_inputs = {
        NodeId("obs"): obs,
        NodeId("target"): target
    }
    
    # Initially grads should be None
    assert all(p.grad is None for p in model.parameters())
    
    results = execute(compiled, initial_inputs, context=ctx)
    
    # After execution, grads should be populated because of the inserted Backward node
    assert any(p.grad is not None for p in model.parameters())
    
    # Check GradBuffer output
    grad_nodes = [nid for nid, n in compiled.nodes.items() if n.node_type == "GradBuffer"]
    assert grad_nodes
    grads = results[grad_nodes[0]]
    if isinstance(grads, Value):
        grads = grads.data
    assert grads is not None
    assert isinstance(grads, torch.Tensor)
    assert grads.shape[0] == sum(p.numel() for p in model.parameters())
