import pytest
from core.graph import (
    Graph,
    NODE_TYPE_REPLAY_QUERY,
    NODE_TYPE_TARGET_SYNC,
)
from compiler.passes.validate_handles import validate_handles
from compiler.validation import SEVERITY_ERROR

pytestmark = pytest.mark.unit


def test_handles_valid_passes() -> None:
    """Verifies that correctly registered model and buffer handles pass."""
    g = Graph()
    g.add_node("q_net", "QNetwork", params={"model_handle": "online_q"})
    g.add_node(
        "sync",
        NODE_TYPE_TARGET_SYNC,
        params={"source_handle": "online_q", "target_handle": "target_q"},
    )
    g.add_node("replay", NODE_TYPE_REPLAY_QUERY, params={"buffer_id": "main"})

    report = validate_handles(
        g, model_handles={"online_q", "target_q"}, buffer_handles={"main"}
    )
    assert not report.has_errors()


def test_handles_unknown_model_fails() -> None:
    """Rule H001: Unknown model handle should trigger a compilation error."""
    g = Graph()
    g.add_node(
        "sync",
        NODE_TYPE_TARGET_SYNC,
        params={
            "source_handle": "online_q",
            "target_handle": "target_qq",  # Typo in handle name
        },
    )

    report = validate_handles(g, model_handles={"online_q", "target_q"})
    assert report.has_errors()
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(i.code == "H001" and "target_qq" in i.message for i in issues)


def test_handles_unknown_buffer_fails() -> None:
    """Rule H002: Unknown buffer handle should trigger a compilation error."""
    g = Graph()
    g.add_node("replay", NODE_TYPE_REPLAY_QUERY, params={"buffer_id": "missing_buffer"})

    report = validate_handles(g, buffer_handles={"main"})
    assert report.has_errors()
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(i.code == "H002" and "missing_buffer" in i.message for i in issues)


def test_handles_default_buffer_fails_if_not_present() -> None:
    """Verifies that the default 'main' buffer is checked even if not explicitly provided."""
    g = Graph()
    g.add_node("replay", NODE_TYPE_REPLAY_QUERY)  # Implicitly uses 'main'

    report = validate_handles(g, buffer_handles={"other_buffer"})
    assert report.has_errors()
    assert any(i.code == "H002" and "main" in i.message for i in report.issues)
