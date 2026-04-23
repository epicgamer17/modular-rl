import pytest
from core.graph import Graph
from runtime.specs import (
    register_spec,
    OperatorSpec,
    SingleObs,
    BatchObs,
    SingleQ,
    BatchQ,
)
from compiler.passes.validate_ports import validate_ports
from compiler.validation import SEVERITY_ERROR

pytestmark = pytest.mark.unit

# Register specific operators for shape/rank validation tests
register_spec(
    "QValuesSingle", OperatorSpec.create(name="QValuesSingle", inputs={"obs": SingleObs}, outputs=SingleQ)
)
register_spec(
    "QValuesBatch", OperatorSpec.create(name="QValuesBatch", inputs={"obs": BatchObs}, outputs=BatchQ)
)
register_spec("SingleObsSource", OperatorSpec.create(name="SingleObsSource", inputs={}, outputs=SingleObs))
register_spec("BatchObsSource", OperatorSpec.create(name="BatchObsSource", inputs={}, outputs=BatchObs))


def test_shape_single_to_single_passes() -> None:
    """Correct wiring of rank-1 obs to rank-1 net should pass."""
    g = Graph()
    g.add_node("src", "SingleObsSource")
    g.add_node("net", "QValuesSingle")
    g.add_edge("src", "net", dst_port="obs")

    report = validate_ports(g)
    assert not report.has_errors()


def test_shape_batch_to_batch_passes() -> None:
    """Correct wiring of rank-2 obs to rank-2 net should pass."""
    g = Graph()
    g.add_node("src", "BatchObsSource")
    g.add_node("net", "QValuesBatch")
    g.add_edge("src", "net", dst_port="obs")

    report = validate_ports(g)
    assert not report.has_errors()


def test_shape_single_to_batch_fails() -> None:
    """Wiring a single observation to a batch-expecting node should fail (E204)."""
    g = Graph()
    g.add_node("src", "SingleObsSource")
    g.add_node("net", "QValuesBatch")
    g.add_edge("src", "net", dst_port="obs")

    report = validate_ports(g)
    assert report.has_errors()
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(
        i.code == "E204" and "BatchObs" in i.message and "SingleObs" in i.message
        for i in issues
    )


def test_shape_batch_to_single_fails() -> None:
    """Wiring a batch of observations to a single-expecting node should fail (E204)."""
    g = Graph()
    g.add_node("src", "BatchObsSource")
    g.add_node("net", "QValuesSingle")
    g.add_edge("src", "net", dst_port="obs")

    report = validate_ports(g)
    assert report.has_errors()
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(
        i.code == "E204" and "SingleObs" in i.message and "BatchObs" in i.message
        for i in issues
    )
