import pytest
import torch
from core.graph import Graph, NODE_TYPE_SINK
from runtime.registry import PortSpec, OperatorSpec, register_spec, SingleObs, ScalarLoss
from runtime.executor import execute, register_operator
from runtime.refs import Value
from runtime.signals import NoOp
from compiler.pipeline import compile_graph

pytestmark = pytest.mark.unit


def test_optional_port_missing_passes() -> None:
    """Verifies that missing an optional port does not trigger E205."""
    register_spec(
        "OptPortOp",
        OperatorSpec.create(
            name="OptPortOp",
            inputs={
                "required_in": PortSpec(spec=SingleObs, required=True),
                "optional_in": PortSpec(spec=SingleObs, required=False),
            },
            outputs=SingleObs
        ),
    )
    register_spec("OptSrc", OperatorSpec.create(name="OptSrc", outputs=SingleObs))
    register_spec("OptSink", OperatorSpec.create(name="OptSink", inputs={"in": SingleObs}))

    g = Graph()
    g.add_node("s", "OptSrc")
    g.add_node("op", "OptPortOp")
    g.add_node("sink", "OptSink")

    g.add_edge("s", "op", dst_port="required_in")
    g.add_edge("op", "sink")

    # Missing optional_in should pass compilation
    compile_graph(g)


def test_required_port_missing_fails() -> None:
    """Verifies that missing a required port triggers E205."""
    register_spec(
        "ReqPortOp",
        OperatorSpec.create(
            name="ReqPortOp",
            inputs={"required_in": PortSpec(spec=SingleObs, required=True)},
        ),
    )

    g = Graph()
    g.add_node("op", "ReqPortOp")

    # Missing required_in should fail compilation
    with pytest.raises(RuntimeError, match="E205"):
        compile_graph(g)


def test_default_value_injection() -> None:
    """Verifies that default values are injected at runtime if port is missing."""
    DEFAULT_VAL = 0.5

    def op_with_default(node, inputs, context=None):
        val = inputs.get("epsilon")
        return val

    register_operator("DefaultOp", op_with_default)
    register_spec(
        "DefaultOp",
        OperatorSpec.create(
            name="DefaultOp",
            inputs={"epsilon": PortSpec(spec=SingleObs, required=False, default=DEFAULT_VAL)},
        ),
    )

    g = Graph()
    g.add_node("op", "DefaultOp")

    # Execute with no inputs connected to 'op'
    outputs = execute(g, {})
    
    # The default value should have been injected and returned by the operator
    assert "op" in outputs
    res = outputs["op"]
    if isinstance(res, Value):
        res = res.data
    assert res == DEFAULT_VAL
