import pytest

from compiler.pipeline import compile_graph
from core.graph import Graph
from runtime.registry import OperatorSpec, Scalar, clear_registry, register_base_specs, register_spec

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def setup_specs():
    clear_registry()
    register_base_specs()
    register_spec(
        "Optimizer",
        OperatorSpec.create(
            name="Optimizer",
            inputs={"loss": Scalar("float32")},
            outputs={"done": Scalar("bool")},
            allowed_contexts={"learner"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=True,
            updates_params=True,
            parameter_handles=["model_handle", "optimizer_handle"],
        ),
    )


def test_double_optimizer_rejected():
    g = Graph()
    g.add_node("pred", "Source")
    g.add_node("target", "Source")
    g.add_node("loss", "MSELoss")
    g.add_edge("pred", "loss", dst_port="pred")
    g.add_edge("target", "loss", dst_port="target")

    g.add_node(
        "opt1",
        "Optimizer",
        params={"model_handle": "online_q", "optimizer_handle": "opt_a"},
    )
    g.add_node(
        "opt2",
        "Optimizer",
        params={"model_handle": "online_q", "optimizer_handle": "opt_b"},
    )
    g.add_edge("loss", "opt1", dst_port="loss")
    g.add_edge("loss", "opt2", dst_port="loss")

    with pytest.raises(RuntimeError, match="G003"):
        compile_graph(g, context="learner", model_handles={"online_q"})
