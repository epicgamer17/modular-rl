import pytest
from core.graph import Graph, NODE_TYPE_SOURCE, NODE_TYPE_SINK, Edge
from runtime.specs import register_spec, OperatorSpec
from compiler.compiler import compile_graph

pytestmark = pytest.mark.unit

# Register dummy specs to satisfy validate_metadata
register_spec(NODE_TYPE_SOURCE, OperatorSpec.create(name=NODE_TYPE_SOURCE))
register_spec(NODE_TYPE_SINK, OperatorSpec.create(name=NODE_TYPE_SINK))
register_spec("SomeType", OperatorSpec.create(name="SomeType"))


def test_compile_strict_mode_converts_warnings_to_errors() -> None:
    """Verifies that strict=True causes warnings to raise a RuntimeError."""
    g = Graph()
    g.add_node("source", NODE_TYPE_SOURCE)
    g.add_node("sink", NODE_TYPE_SINK)
    # Ensure source reaches sink to avoid E003 (Sinkless Branch)
    g.add_edge("source", "sink")

    # Add an unreachable node to trigger a warning (W001)
    g.add_node("unreachable", "SomeType")

    # Normal mode: passes even with warnings
    compile_graph(g, strict=False)

    # Strict mode: fails because of the unreachable node warning
    with pytest.raises(
        RuntimeError, match="Graph compilation failed in STRICT mode due to warnings"
    ):
        compile_graph(g, strict=True)


def test_compile_normal_mode_allows_warnings() -> None:
    """Verifies that normal mode allows graphs with warnings to compile."""
    g = Graph()
    g.add_node("source", NODE_TYPE_SOURCE)
    g.add_node("sink", NODE_TYPE_SINK)
    g.add_edge("source", "sink")
    g.add_node("unreachable", "SomeType")

    # Should not raise in normal mode
    result = compile_graph(g, strict=False)
    assert isinstance(result, Graph)


def test_compile_hard_error_always_fails() -> None:
    """Verifies that severe errors trigger a failure regardless of the strict flag."""
    g = Graph()
    g.add_node("source", NODE_TYPE_SOURCE)
    g.add_node("sink", NODE_TYPE_SINK)

    # Manually append an edge to a missing node to bypass Graph.add_edge assertions
    # This triggers E001 (Hard Error)
    g.edges.append(Edge(src="source", dst="missing_node"))

    # Both modes should fail because of the hard error
    with pytest.raises(RuntimeError, match="failed with errors"):
        compile_graph(g, strict=False)

    with pytest.raises(RuntimeError, match="failed with errors"):
        compile_graph(g, strict=True)
