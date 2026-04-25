import pytest
from core.graph import Graph
from runtime.registry import register_spec, OperatorSpec, SingleObs, SingleQ, PortSpec, clear_registry
from compiler.pipeline import compile_graph
from compiler.passes.optimization.parameters import collect_trainable_parameters

pytestmark = pytest.mark.unit

def setup_function():
    clear_registry()

def test_collect_trainable_parameters_basic():
    """Verifies that parameter handles are correctly collected from nodes."""
    register_spec(
        "QNet",
        OperatorSpec.create(
            name="QNet",
            inputs={"obs": SingleObs},
            outputs={"q": SingleQ},
            parameter_handles=["model_handle"],
            # Required trainability metadata
            differentiable=True,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            allowed_contexts={"actor", "learner"}
        ),
    )

    g = Graph()
    g.add_node("n1", "QNet", params={"model_handle": "online_q"})
    g.add_node("n2", "QNet", params={"model_handle": "target_q"})
    g.add_node("n3", "QNet", params={"model_handle": "online_q"})

    params = collect_trainable_parameters(g)
    
    assert "online_q" in params
    assert "target_q" in params
    assert set(params["online_q"]) == {"n1", "n3"}
    assert params["target_q"] == ["n2"]

def test_compiler_populates_parameters():
    """Verifies that compile_graph populates the graph's parameters field."""
    register_spec(
        "QNet",
        OperatorSpec.create(
            name="QNet",
            inputs={"obs": PortSpec(spec=SingleObs, required=False)},
            outputs={"q": SingleQ},
            parameter_handles=["model_handle"],
            differentiable=True,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            allowed_contexts={"actor", "learner"}
        ),
    )

    g = Graph()
    g.add_node("n1", "QNet", params={"model_handle": "online_q"})
    
    # We need to provide dummy handles for validation
    compiled_g = compile_graph(g, model_handles={"online_q"})
    
    assert "online_q" in compiled_g.parameters
    assert compiled_g.parameters["online_q"] == ["n1"]

def test_collect_no_parameters():
    """Verifies that an empty dict is returned if no parameters are referenced."""
    register_spec(
        "PureOp",
        OperatorSpec.create(
            name="PureOp",
            inputs={},
            outputs={},
            pure=True,
            allowed_contexts={"actor"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        ),
    )
    
    g = Graph()
    g.add_node("n1", "PureOp")
    
    params = collect_trainable_parameters(g)
    assert params == {}
