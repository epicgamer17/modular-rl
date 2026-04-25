import pytest
from core.graph import Graph, NODE_TYPE_SOURCE, NODE_TYPE_SINK, Edge
from runtime.registry import register_spec, OperatorSpec
from compiler.pipeline import compile_graph

pytestmark = pytest.mark.unit

@pytest.fixture(autouse=True)
def setup_specs() -> None:
    """Register specifications for test nodes. Clears registry for isolation."""
    from runtime.registry import clear_registry
    clear_registry()
    # Register dummy specs to satisfy validate_metadata
    register_spec("StrictSource", OperatorSpec.create(
        name="StrictSource",
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("StrictSink", OperatorSpec.create(
        name="StrictSink",
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("StrictType", OperatorSpec.create(
        name="StrictType",
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))


def test_compile_strict_mode_converts_warnings_to_errors() -> None:
    """Verifies that strict=True causes warnings to raise a RuntimeError."""
    g = Graph()
    g.add_node("source", "StrictSource")
    g.add_node("sink", "StrictSink")
    # Ensure source reaches sink to avoid E003 (Sinkless Branch)
    g.add_edge("source", "sink")

    # Add an unreachable node to trigger a warning (W001)
    g.add_node("unreachable", "StrictType")

    # Normal mode: passes even with warnings
    compile_graph(g, strict=False)

    # Strict mode: fails because of the unreachable node warning
    # Note: Match the specific warning-to-error conversion message
    with pytest.raises(
        RuntimeError, match="Graph compilation failed in STRICT mode due to warnings"
    ):
        compile_graph(g, strict=True)


def test_compile_normal_mode_allows_warnings() -> None:
    """Verifies that normal mode allows graphs with warnings to compile."""
    g = Graph()
    g.add_node("source", "StrictSource")
    g.add_node("sink", "StrictSink")
    g.add_edge("source", "sink")
    g.add_node("unreachable", "StrictType")

    # Should not raise in normal mode
    result = compile_graph(g, strict=False)
    assert isinstance(result, Graph)


def test_compile_hard_error_always_fails() -> None:
    """Verifies that severe errors trigger a failure regardless of the strict flag."""
    g = Graph()
    g.add_node("source", "StrictSource")
    g.add_node("sink", "StrictSink")

    # Manually append an edge to a missing node to bypass Graph.add_edge assertions
    # This triggers E001 (Hard Error)
    g.edges.append(Edge(src="source", dst="missing_node"))

    # Both modes should fail because of the hard error
    with pytest.raises(RuntimeError, match="failed with errors"):
        compile_graph(g, strict=False)

    with pytest.raises(RuntimeError, match="failed with errors"):
        compile_graph(g, strict=True)
