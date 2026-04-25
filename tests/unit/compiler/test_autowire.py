import pytest
from core.graph import Graph
from runtime.registry import register_spec, OperatorSpec, SingleObs, BatchObs, PortSpec
from compiler.passes.structural.ports import validate_ports
from compiler.validation import SEVERITY_ERROR

pytestmark = pytest.mark.unit


def test_autowire_one_valid_passes() -> None:
    """Verifies that if only one port is compatible, it auto-selects."""
    register_spec(
        "DualInput",
        OperatorSpec.create(
            name="DualInput", 
            inputs={
                "obs": SingleObs, 
                "other": PortSpec(spec=BatchObs, required=False)
            },
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        ),
    )
    register_spec("ObsSrc", OperatorSpec.create(name="ObsSrc", outputs=SingleObs, differentiable=False, creates_grad=False, consumes_grad=False, updates_params=False))

    g = Graph()
    g.add_node("s", "ObsSrc")
    g.add_node("d", "DualInput")
    g.add_edge("s", "d")  # No dst_port specified, should auto-wire to 'obs'

    report = validate_ports(g)
    assert not report.has_errors(), f"Found validation errors: {[f'[{i.code}] {i.message}' for i in report.issues]}"


def test_autowire_ambiguous_fails() -> None:
    """Verifies that multiple compatible ports trigger E206."""
    register_spec(
        "MultiInput",
        OperatorSpec.create(name="MultiInput", inputs={"in1": SingleObs, "in2": SingleObs}, differentiable=False, creates_grad=False, consumes_grad=False, updates_params=False),
    )
    register_spec("Src", OperatorSpec.create(name="Src", outputs=SingleObs, differentiable=False, creates_grad=False, consumes_grad=False, updates_params=False))

    g = Graph()
    g.add_node("s", "Src")
    g.add_node("d", "MultiInput")
    g.add_edge("s", "d")

    report = validate_ports(g)
    assert report.has_errors()
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(i.code == "E206" and "in1" in i.message and "in2" in i.message for i in issues)


def test_typo_suggestion() -> None:
    """Verifies that a typo in dst_port suggests a compatible port (E203)."""
    register_spec(
        "Target", OperatorSpec.create(name="Target", inputs={"correct_port": SingleObs}, differentiable=False, creates_grad=False, consumes_grad=False, updates_params=False)
    )
    register_spec("Src", OperatorSpec.create(name="Src", outputs=SingleObs, differentiable=False, creates_grad=False, consumes_grad=False, updates_params=False))

    g = Graph()
    g.add_node("s", "Src")
    g.add_node("d", "Target")
    g.add_edge("s", "d", dst_port="wrong_port")

    report = validate_ports(g)
    assert report.has_errors()
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any("Did you mean dst_port='correct_port'?" in i.message for i in issues)


def test_mismatch_suggestion() -> None:
    """Verifies that a type mismatch suggests a compatible port if available (E204)."""
    register_spec(
        "MismatchTarget",
        OperatorSpec.create(
            name="MismatchTarget", inputs={"obs": BatchObs, "single_obs": SingleObs}, differentiable=False, creates_grad=False, consumes_grad=False, updates_params=False
        ),
    )
    register_spec("SingleSrc", OperatorSpec.create(name="SingleSrc", outputs=SingleObs, differentiable=False, creates_grad=False, consumes_grad=False, updates_params=False))

    g = Graph()
    g.add_node("s", "SingleSrc")
    g.add_node("d", "MismatchTarget")
    # Manually connecting to 'obs' (BatchObs) while source is SingleObs
    g.add_edge("s", "d", dst_port="obs")

    report = validate_ports(g)
    assert report.has_errors()
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any("Suggestion: Use dst_port='single_obs'" in i.message for i in issues)
