import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn.specs import register_dqn_specs
from core.graph import Graph, NodeId, EdgeType
from runtime.context import ExecutionContext
from runtime.executor import execute, register_operator
from runtime.operators.losses import register_loss_operators
from runtime.specs import OperatorSpec, clear_registry, register_base_specs, register_spec
from runtime.state import GradientRegistry, OptimizerState

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def setup_specs():
    clear_registry()
    register_base_specs()
    register_dqn_specs()
    register_loss_operators()


def _register_forward():
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
            parameter_handles=["model_handle"],
        ),
    )


def test_grad_sum_matches_manual():
    _register_forward()

    graph = Graph()
    graph.add_node("obs", "Source")
    graph.add_node("target", "Source")
    graph.add_node("forward", "TestForward", params={"model_handle": "online_q"})
    graph.add_node("mse", "MSELoss")
    graph.add_node(
        "backward",
        "Backward",
        params={"model_handle": "online_q", "optimizer_handle": "main_opt"},
    )
    graph.add_node("accumulate", "AccumulateGrad", params={"model_handle": "online_q", "k": 4})

    graph.add_edge("obs", "forward", dst_port="obs")
    graph.add_edge("forward", "mse", src_port="default", dst_port="pred")
    graph.add_edge("target", "mse", dst_port="target")
    graph.add_edge("mse", "backward", dst_port="loss")
    graph.add_edge("backward", "accumulate", edge_type=EdgeType.CONTROL)

    model = nn.Linear(2, 1, bias=True)
    manual_model = nn.Linear(2, 1, bias=True)
    manual_model.load_state_dict(model.state_dict())

    ctx = ExecutionContext()
    ctx.model_registry.register("online_q", model)
    ctx.optimizer_registry.register(
        "main_opt",
        OptimizerState(optim.SGD(model.parameters(), lr=0.01)),
    )

    batches = [
        (torch.tensor([[1.0, -2.0]]), torch.tensor([[0.5]])),
        (torch.tensor([[0.0, 3.0]]), torch.tensor([[-1.0]])),
        (torch.tensor([[2.0, 1.0]]), torch.tensor([[1.5]])),
        (torch.tensor([[-1.0, 4.0]]), torch.tensor([[0.25]])),
    ]

    manual_sum = None
    for obs, target in batches:
        manual_model.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(manual_model(obs), target)
        loss.backward()
        flat = GradientRegistry.flatten_model_grads(manual_model)
        manual_sum = flat if manual_sum is None else manual_sum + flat

        execute(
            graph,
            {
                NodeId("obs"): obs,
                NodeId("target"): target,
            },
            context=ctx,
        )

    accumulated = ctx.get_gradients("online_q")
    assert accumulated is not None
    assert ctx.gradient_registry.count("online_q") == 4
    assert torch.allclose(accumulated, manual_sum, atol=1e-6)
