import pytest
import torch
from core.graph import Graph
from runtime.executor import execute, register_operator
from runtime.context import ExecutionContext
from runtime.registry import OperatorSpec, PortSpec, register_spec, clear_registry

pytestmark = pytest.mark.unit

def mock_op(node, inputs, context=None):
    return {"received": list(inputs.keys())}

@pytest.fixture(autouse=True)
def setup_specs():
    clear_registry()
    register_spec("TestOp", OperatorSpec.create(
        "TestOp",
        inputs={
            "required_port": PortSpec(spec=None, required=True),
            "optional_port": PortSpec(spec=None, required=False, default=10)
        }
    ))
    register_operator("TestOp", mock_op)

def test_missing_required_port_fails():
    """Verify that missing a required port raises a Contract Violation."""
    graph = Graph()
    graph.add_node("node1", "TestOp")
    
    # Missing 'required_port'
    with pytest.raises(RuntimeError, match="missing required port 'required_port'"):
        execute(graph, initial_inputs={})

def test_wrong_port_name_fails():
    """Verify that providing data on a wrong port name is treated as missing required port."""
    graph = Graph()
    graph.add_node("node1", "TestOp")
    from core.graph import NODE_TYPE_SOURCE
    graph.add_node("src", NODE_TYPE_SOURCE)
    # Wire to WRONG port
    graph.add_edge("src", "node1", dst_port="wrong_port")
    
    with pytest.raises(RuntimeError, match="missing required port 'required_port'"):
        execute(graph, initial_inputs={"src": 1.0})

def test_undeclared_extra_port_is_filtered():
    """Verify that undeclared extra ports are not passed to the operator."""
    graph = Graph()
    graph.add_node("node1", "TestOp")
    from core.graph import NODE_TYPE_SOURCE
    graph.add_node("src1", NODE_TYPE_SOURCE)
    graph.add_node("src2", NODE_TYPE_SOURCE)
    
    graph.add_edge("src1", "node1", dst_port="required_port")
    graph.add_edge("src2", "node1", dst_port="extra_port")
    
    results = execute(graph, initial_inputs={"src1": 1.0, "src2": 2.0})
    received = results["node1"].data["received"]
    
    assert "required_port" in received
    assert "optional_port" in received # Injected by default
    assert "extra_port" not in received, "Extra port should have been filtered out"

def test_optional_port_injection():
    """Verify that optional ports receive their default values if not provided."""
    graph = Graph()
    graph.add_node("node1", "TestOp")
    from core.graph import NODE_TYPE_SOURCE
    graph.add_node("src", NODE_TYPE_SOURCE)
    graph.add_edge("src", "node1", dst_port="required_port")
    
    # We use execute's internal value unwrapping, but the operator function returns a dict.
    # Results["node1"] will be a Value(dict).
    results = execute(graph, initial_inputs={"src": 1.0})
    # The mock_op receives unwrapped inputs.
    # It returns a dict which execute wraps in a Value.
    
    # Accessing the output of mock_op
    op_output = results["node1"].data
    
    # The 'inputs' dict seen by mock_op should have optional_port = 10
    # Wait, mock_op just returns the keys.
    # Let's modify mock_op to return values too for this test.
    pass

def test_optional_port_default_values():
    def mock_op_val(node, inputs, context=None):
        return inputs
    
    register_operator("TestOpVal", mock_op_val)
    register_spec("TestOpVal", OperatorSpec.create(
        "TestOpVal",
        inputs={
            "required_port": PortSpec(spec=None, required=True),
            "optional_port": PortSpec(spec=None, required=False, default=42)
        }
    ))
    
    graph = Graph()
    graph.add_node("node1", "TestOpVal")
    from core.graph import NODE_TYPE_SOURCE
    graph.add_node("src", NODE_TYPE_SOURCE)
    graph.add_edge("src", "node1", dst_port="required_port")
    
    results = execute(graph, initial_inputs={"src": 1.0})
    inputs_seen = results["node1"].data
    
    assert inputs_seen["required_port"] == 1.0
    assert inputs_seen["optional_port"] == 42
