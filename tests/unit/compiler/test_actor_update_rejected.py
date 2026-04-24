import pytest

from compiler.compiler import compile_graph
from core.graph import Graph
from runtime.specs import OperatorSpec, Scalar, clear_registry, register_base_specs, register_spec

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def setup_specs():
    clear_registry()
    register_base_specs()
    register_spec(
        "InferenceUpdate",
        OperatorSpec.create(
            name="InferenceUpdate",
            inputs={},
            outputs={"done": Scalar("bool")},
            allowed_contexts={"actor", "learner"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=True,
            parameter_handles=["model_handle"],
        ),
    )


def test_actor_update_rejected():
    g = Graph()
    g.add_node("update", "InferenceUpdate", params={"model_handle": "online_q"})

    with pytest.raises(RuntimeError, match="G004"):
        compile_graph(g, context="actor", model_handles={"online_q"})


def test_actor_gradient_node_rejected():
    g = Graph()
    g.add_node("pred", "Source")
    g.add_node("target", "Source")
    g.add_node("loss", "MSELoss")
    g.add_edge("pred", "loss", dst_port="pred")
    g.add_edge("target", "loss", dst_port="target")

    with pytest.raises(RuntimeError, match="G005"):
        compile_graph(g, context="actor")
