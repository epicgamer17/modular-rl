from runtime.bootstrap import bootstrap_runtime
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn.specs import register_dqn_specs
from compiler.pipeline import compile_graph
from core.graph import Graph, NodeId
from runtime.context import ExecutionContext
from runtime.executor import execute, register_operator
from runtime.registry import OperatorSpec, clear_registry, register_spec
from runtime.state import OptimizerState
from runtime.refs import Value

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def setup_specs():
    clear_registry()
    bootstrap_runtime()
    register_dqn_specs()


def test_grad_buffer_written():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 1)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()

    def op_forward(node, inputs, context):
        return context.get_model(node.params["model_handle"])(inputs["obs"])

    register_operator("TestForward", op_forward)
    register_spec(
        "TestForward",
        OperatorSpec.create(
            name="TestForward",
            inputs={"obs": None},
            outputs={"pred": None},
            differentiable=True,
            creates_grad=True,
            consumes_grad=False,
            updates_params=False,
            parameter_handles=["model_handle"],
        ),
    )

    def op_test_optimizer(node, inputs, context):
        context.get_optimizer(node.params["optimizer_handle"]).step()
        return True

    register_operator("TestOptimizer", op_test_optimizer)
    register_spec(
        "TestOptimizer",
        OperatorSpec.create(
            name="TestOptimizer",
            inputs={"loss": None},
            outputs={"done": None},
            differentiable=False,
            creates_grad=False,
            consumes_grad=True,
            updates_params=True,
            parameter_handles=["model_handle", "optimizer_handle"],
        ),
    )

    g = Graph()
    g.add_node("obs", "Source")
    g.add_node("forward", "TestForward", params={"model_handle": "online_q"})
    g.add_edge("obs", "forward", dst_port="obs")
    g.add_node("target", "Source")
    g.add_node("mse", "MSELoss")
    g.add_edge("forward", "mse", src_port="default", dst_port="pred")
    g.add_edge("target", "mse", dst_port="target")
    g.add_node(
        "opt",
        "TestOptimizer",
        params={"model_handle": "online_q", "optimizer_handle": "main_opt"},
    )
    g.add_edge("mse", "opt", dst_port="loss")

    compiled = compile_graph(g, context="learner", model_handles={"online_q"})

    ctx = ExecutionContext()
    ctx.model_registry.register("online_q", model)
    ctx.optimizer_registry.register(
        "main_opt",
        OptimizerState(optim.SGD(model.parameters(), lr=0.01)),
    )

    results = execute(
        compiled,
        {
            NodeId("obs"): torch.randn(2, 4),
            NodeId("target"): torch.randn(2, 1),
        },
        context=ctx,
    )

    stored_grads = ctx.get_gradients("online_q")
    assert stored_grads is not None
    assert stored_grads.ndim == 1
    assert stored_grads.shape[0] == sum(p.numel() for p in model.parameters())
    assert any(param.grad is not None for param in model.parameters())

    grad_node = next(nid for nid, node in compiled.nodes.items() if node.node_type == "GradBuffer")
    grad_output = results[grad_node]
    if isinstance(grad_output, Value):
        grad_output = grad_output.data
    assert torch.allclose(grad_output, stored_grads)
