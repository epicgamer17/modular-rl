import pytest
import torch
from core.graph import Graph
from core.schema import Schema, Field, TensorSpec
from runtime.specs import (
    register_spec,
    OperatorSpec,
    BatchObs,
    SingleObs,
    TransitionBatch,
    ScalarLoss,
)
from compiler.passes.validate_ports import validate_ports
from compiler.validation import SEVERITY_ERROR

pytestmark = pytest.mark.unit


def test_explainable_port_mismatch_e204() -> None:
    """Rule E204: Port mismatch should show connection path and Expected/Got blocks."""
    # Use unique names to avoid registry conflicts
    register_spec(
        "ExpSampler", OperatorSpec.create(name="ExpSampler", inputs={}, outputs=TransitionBatch)
    )
    register_spec(
        "ExpQNet", OperatorSpec.create(name="ExpQNet", inputs={"obs": SingleObs}, outputs=BatchObs)
    )

    g = Graph()
    g.add_node("sampler", "ExpSampler")
    g.add_node("q_net", "ExpQNet")
    # TransitionBatch (Schema) -> SingleObs (Tensor)
    g.add_edge("sampler", "q_net", dst_port="obs")

    report = validate_ports(g)
    assert report.has_errors()
    issue = report.get_issues_by_severity(SEVERITY_ERROR)[0]

    # Verify high-quality diagnostic output
    assert "q_net.obs <- sampler" in issue.message
    assert "Expected:" in issue.message
    assert "Got:" in issue.message
    # format_spec(SingleObs) should show tags if present
    assert "single" in issue.message.lower()
    # format_spec(TransitionBatch) should show fields
    assert "TransitionBatch" in issue.message
    assert "obs:" in issue.message
    assert "reward:" in issue.message


def test_explainable_field_mismatch_e311() -> None:
    """Rule E311: Field mismatch should show field details and path."""
    # Create a schema that is ALMOST TransitionBatch but with float32 action
    BadBatch = Schema(
        fields=[
            Field("obs", BatchObs),
            Field("action", TensorSpec(shape=(-1,), dtype="float32")),  # WRONG DTYPE (should be int64)
            Field("reward", TensorSpec(shape=(-1,), dtype="float32")),
            Field("next_obs", BatchObs),
            Field("done", TensorSpec(shape=(-1,), dtype="bool")),
        ]
    )

    register_spec("BadSampler", OperatorSpec.create(name="BadSampler", inputs={}, outputs=BadBatch))
    register_spec(
        "ExpTDLoss",
        OperatorSpec.create(name="ExpTDLoss", inputs={"batch": TransitionBatch}, outputs=ScalarLoss),
    )

    g = Graph()
    g.add_node("sampler", "BadSampler")
    g.add_node("loss", "ExpTDLoss")
    g.add_edge("sampler", "loss", dst_port="batch")

    report = validate_ports(g)
    assert report.has_errors()
    # Find the field mismatch error
    issue = [i for i in report.issues if i.code == "E311"][0]

    assert "loss.batch <- sampler" in issue.message
    assert "Field 'action' mismatch:" in issue.message
    assert "Expected: int64" in issue.message
    assert "Got:      float32" in issue.message


def test_explainable_missing_field_e310() -> None:
    """Rule E310: Missing field should show path and missing field info."""
    # Schema missing the 'reward' field
    IncompleteBatch = Schema(
        fields=[
            Field("obs", BatchObs),
            Field("action", TensorSpec(shape=(-1,), dtype="int64")),
            Field("next_obs", BatchObs),
            Field("done", TensorSpec(shape=(-1,), dtype="bool")),
        ]
    )

    register_spec(
        "IncompleteSampler", OperatorSpec.create(name="IncompleteSampler", inputs={}, outputs=IncompleteBatch)
    )

    g = Graph()
    g.add_node("sampler", "IncompleteSampler")
    g.add_node("loss", "ExpTDLoss")
    g.add_edge("sampler", "loss", dst_port="batch")

    report = validate_ports(g)
    assert report.has_errors()
    issue = [i for i in report.issues if i.code == "E310"][0]

    assert "loss.batch <- sampler" in issue.message
    assert "Field 'reward' missing from schema" in issue.message
    assert "Expected in batch:" in issue.message
