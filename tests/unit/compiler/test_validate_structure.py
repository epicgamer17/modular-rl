import pytest
from core.graph import (
    Graph,
    NodeId,
    NODE_TYPE_SOURCE,
    NODE_TYPE_SINK,
    Edge,
    Node,
    Schema,
)
from compiler.passes.validate_structure import validate_structure
from compiler.validation import SEVERITY_ERROR, SEVERITY_WARN

pytestmark = pytest.mark.unit


def test_validate_missing_node() -> None:
    """Verifies that edges referencing non-existent nodes are caught (E001)."""
    g = Graph()
    # Manually inject nodes and edges to bypass Graph.add_node/add_edge assertions
    # which would prevent creating a broken graph for testing.
    g.nodes[NodeId("a")] = Node(
        node_id=NodeId("a"),
        node_type=NODE_TYPE_SOURCE,
        schema_in=Schema(fields=[]),
        schema_out=Schema(fields=[]),
    )
    g.edges.append(Edge(src=NodeId("a"), dst=NodeId("missing")))

    report = validate_structure(g)
    assert report.has_errors(), "Missing node should be an error"
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(
        i.code == "E001" and "missing" in i.message for i in issues
    ), "Expected E001 for missing node"


def test_validate_cycle() -> None:
    """Verifies that cycles in the graph are caught (E002)."""
    g = Graph()
    g.add_node("a", NODE_TYPE_SOURCE)
    g.add_node("b", "Transform")
    g.add_node("sink", NODE_TYPE_SINK)

    g.add_edge("a", "b")
    g.add_edge("b", "a")  # Cycle
    g.add_edge("b", "sink")

    report = validate_structure(g)
    assert report.has_errors(), "Cycles should be an error"
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(i.code == "E002" for i in issues), "Expected E002 for cycles"


def test_validate_disconnected_node() -> None:
    """Verifies that unreachable nodes are reported as warnings (W001)."""
    g = Graph()
    g.add_node("a", NODE_TYPE_SOURCE)
    g.add_node("sink", NODE_TYPE_SINK)
    g.add_node("c", "Transform")  # Disconnected from SOURCE

    g.add_edge("a", "sink")

    report = validate_structure(g)
    # Unreachable nodes are warnings, not errors
    assert not report.has_errors()
    warnings = report.get_issues_by_severity(SEVERITY_WARN)
    assert any(
        i.code == "W001" and i.node_id == "c" for i in warnings
    ), "Expected W001 for unreachable node 'c'"


def test_validate_sinkless_branch() -> None:
    """Verifies that branches that don't reach a SINK are caught (E003)."""
    g = Graph()
    g.add_node("source", NODE_TYPE_SOURCE)
    g.add_node("dangling", "Transform")
    # No SINK reachable from dangling

    g.add_edge("source", "dangling")

    report = validate_structure(g)
    assert report.has_errors(), "Sinkless branches should be an error"
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(
        i.code == "E003" and i.node_id == "dangling" for i in issues
    ), "Expected E003 for node 'dangling'"


def test_validate_valid_graph() -> None:
    """Verifies that a correct graph passes validation."""
    g = Graph()
    g.add_node("source", NODE_TYPE_SOURCE)
    g.add_node("mid", "Transform")
    g.add_node("sink", NODE_TYPE_SINK)

    g.add_edge("source", "mid")
    g.add_edge("mid", "sink")

    report = validate_structure(g)
    assert not report.has_errors(), "Valid graph should not have errors"
    assert len(report.issues) == 0, "Valid graph should have zero issues"
