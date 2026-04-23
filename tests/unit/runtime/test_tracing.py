import pytest
from core.graph import Graph, NODE_TYPE_SOURCE
from runtime.executor import execute, register_operator
from runtime.tracing import TraceLogger
from runtime.context import ExecutionContext
from runtime.values import Skipped, Value

pytestmark = pytest.mark.unit


def test_execution_tracing() -> None:
    """Verifies that every node emits a trace with inputs and outputs."""
    register_operator("AddTrace", lambda node, inputs, ctx: inputs["a"] + inputs["b"])

    g = Graph()
    g.add_node("a", NODE_TYPE_SOURCE)
    g.add_node("b", NODE_TYPE_SOURCE)
    g.add_node("sum", "AddTrace")
    g.add_edge("a", "sum", dst_port="a")
    g.add_edge("b", "sum", dst_port="b")

    tracer = TraceLogger()
    initial_inputs = {"a": 10, "b": 20}

    execute(g, initial_inputs, tracer=tracer)

    trace = tracer.get_step(0)
    assert "sum" in trace.nodes
    node_trace = trace.nodes["sum"]
    assert node_trace.inputs == {"a": 10, "b": 20}
    assert node_trace.outputs.data == 30
    assert node_trace.runtime_ms >= 0


def test_tracing_skipped_nodes() -> None:
    """Verifies that skipped nodes are logged with a reason."""

    g = Graph()
    g.add_node("src", NODE_TYPE_SOURCE)
    g.add_node("skip", "SkipOpTrace")
    g.add_edge("src", "skip")

    # Force a skip reason
    register_operator("SkipOpTrace", lambda node, inputs, ctx: Skipped("test_reason"))

    tracer = TraceLogger()
    execute(g, {"src": 1}, tracer=tracer)

    trace = tracer.get_step(0)
    assert "skip" in trace.nodes
    assert trace.nodes["skip"].skipped_reason == "test_reason"
    assert isinstance(trace.nodes["skip"].outputs, Skipped)


def test_replay_deterministic() -> None:
    """Verifies that a trace can be replayed deterministically."""
    register_operator("MulTrace", lambda node, inputs, ctx: inputs["x"] * 2)

    g = Graph()
    g.add_node("x", NODE_TYPE_SOURCE)
    g.add_node("y", "MulTrace")
    g.add_edge("x", "y", dst_port="x")

    tracer = TraceLogger()
    execute(g, {"x": 5}, tracer=tracer)

    trace = tracer.get_step(0)

    # Replay should pass
    ctx = ExecutionContext()
    trace.run(g, ctx)

    # If we change the operator logic, replay should fail
    # We must register a new type because register_operator is global and often cached
    register_operator("MulTrace", lambda node, inputs, ctx: inputs["x"] * 3)
    with pytest.raises(AssertionError, match="Replay mismatch"):
        trace.run(g, ctx)


def test_upstream_skip_tracing() -> None:
    """Verifies that automatic skip propagation is recorded in the trace."""
    register_operator("NeverRun", lambda node, inputs, ctx: 1/0) # Should not run
    
    g = Graph()
    g.add_node("a", NODE_TYPE_SOURCE)
    g.add_node("b", "NeverRun")
    g.add_edge("a", "b", dst_port="in")
    
    tracer = TraceLogger()
    # Initial input is a Skip
    execute(g, {"a": Skipped("initial_skip")}, tracer=tracer)
    
    trace = tracer.get_step(0)
    assert "b" in trace.nodes
    assert trace.nodes["b"].skipped_reason == "upstream_failed_a"
    assert isinstance(trace.nodes["b"].outputs, Skipped)
