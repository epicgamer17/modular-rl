import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from agents.dqn.specs import register_dqn_specs
from core.graph import Graph, NodeId, EdgeType
from runtime.context import ExecutionContext
from runtime.executor import execute, register_operator
from runtime.registry import OperatorSpec, clear_registry, register_base_specs, register_spec
from runtime.state import OptimizerState
from runtime.refs import Value

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def setup_specs():
    clear_registry()
    register_base_specs()
    register_dqn_specs()


def _build_accum_graph():
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
    graph.add_node(
        "step",
        "OptimizerStepEvery",
        params={"model_handle": "online_q", "optimizer_handle": "main_opt", "k": 4},
    )

    graph.add_edge("obs", "forward", dst_port="obs")
    graph.add_edge("forward", "mse", src_port="default", dst_port="pred")
    graph.add_edge("target", "mse", dst_port="target")
    graph.add_edge("mse", "backward", dst_port="loss")
    graph.add_edge("backward", "accumulate", edge_type=EdgeType.CONTROL)
    graph.add_edge("accumulate", "step", edge_type=EdgeType.CONTROL)
    return graph


def test_optimizer_runs_every_k_steps():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)

    ctx = ExecutionContext()
    ctx.model_registry.register("online_q", model)
    ctx.optimizer_registry.register(
        "main_opt",
        OptimizerState(optim.SGD(model.parameters(), lr=0.1)),
    )

    graph = _build_accum_graph()
    step_flags = []
    weights = []

    for _ in range(8):
        outputs = execute(
            graph,
            {
                NodeId("obs"): torch.tensor([[1.0]]),
                NodeId("target"): torch.tensor([[0.0]]),
            },
            context=ctx,
        )
        step_result = outputs["step"]
        if isinstance(step_result, Value):
            step_result = step_result.data
        step_flags.append(step_result["stepped"])
        weights.append(model.weight.item())

    assert step_flags == [False, False, False, True, False, False, False, True]
    assert weights[2] == pytest.approx(1.0)
    assert weights[3] != pytest.approx(weights[2])
    assert weights[6] == pytest.approx(weights[4])
    assert weights[7] != pytest.approx(weights[6])
