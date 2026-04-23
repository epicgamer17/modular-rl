import pytest
from core.graph import Graph
from runtime.specs import (
    register_spec,
    OperatorSpec,
    ObsTensor,
    ActionValuesTensor,
    TransitionBatch,
    ScalarLoss,
)
from compiler.passes.validate_ports import validate_ports
from compiler.validation import SEVERITY_ERROR

pytestmark = pytest.mark.unit

# Setup specifications for test nodes
register_spec(
    "QValuesSingle",
    OperatorSpec.create(inputs={"obs": ObsTensor}, outputs=ActionValuesTensor),
)
register_spec(
    "TDLoss", OperatorSpec.create(inputs={"batch": TransitionBatch}, outputs=ScalarLoss)
)
register_spec("Sampler", OperatorSpec.create(inputs={}, outputs=TransitionBatch))
register_spec("ObservationSource", OperatorSpec.create(inputs={}, outputs=ObsTensor))


def test_ports_correct_obs_wiring() -> None:
    """Verifies that correct wiring between compatible ports passes."""
    g = Graph()
    g.add_node("src", "ObservationSource")
    g.add_node("q_net", "QValuesSingle")
    # Wiring ObservationSource (ObsTensor) -> QValuesSingle.obs (ObsTensor)
    g.add_edge("src", "q_net", dst_port="obs")

    report = validate_ports(g)
    assert not report.has_errors(), f"Correct wiring failed: {report.issues}"


def test_ports_batch_to_obs_fails() -> None:
    """Verifies that wiring a Batch output to a Tensor input fails (E204)."""
    g = Graph()
    g.add_node("sampler", "Sampler")
    g.add_node("q_net", "QValuesSingle")
    # Wiring Sampler (TransitionBatch Schema) -> QValuesSingle.obs (ObsTensor)
    g.add_edge("sampler", "q_net", dst_port="obs")

    report = validate_ports(g)
    assert report.has_errors(), "Incompatible wiring (Schema to Tensor) should fail"
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(
        i.code == "E204" and "obs" in i.message and "TransitionBatch" in i.message
        for i in issues
    )


def test_ports_q_values_to_loss_batch_fails() -> None:
    """Verifies that wiring a Tensor output to a Schema input fails (E204)."""
    g = Graph()
    g.add_node("q_net", "QValuesSingle")
    g.add_node("loss", "TDLoss")
    # Wiring QValuesSingle (ActionValuesTensor) -> TDLoss.batch (TransitionBatch Schema)
    g.add_edge("q_net", "loss", dst_port="batch")

    report = validate_ports(g)
    assert report.has_errors(), "Incompatible wiring (Tensor to Schema) should fail"
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(
        i.code == "E204" and "batch" in i.message and "SingleQ" in i.message
        for i in issues
    )


def test_ports_invalid_port_name() -> None:
    """Verifies that referencing a non-existent port on a node fails (E203)."""
    g = Graph()
    g.add_node("src", "ObservationSource")
    g.add_node("q_net", "QValuesSingle")
    # QValuesSingle only has "obs", not "wrong_port"
    g.add_edge("src", "q_net", dst_port="wrong_port")

    report = validate_ports(g)
    assert report.has_errors()
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(i.code == "E203" and "wrong_port" in i.message for i in issues)
