import pytest
from core.graph import Graph
from core.schema import Schema, Field
from runtime.registry import (
    register_spec,
    OperatorSpec,
    TransitionBatch,
    BatchObs,
    Tensor,
)
from compiler.passes.structural.ports import validate_ports
from compiler.validation import SEVERITY_ERROR

pytestmark = pytest.mark.unit

@pytest.fixture(autouse=True)
def setup_specs() -> None:
    """Register specifications for test nodes. Clears registry for isolation."""
    from runtime.registry import clear_registry
    clear_registry()
    # Define TDLoss to expect the standard TransitionBatch
    register_spec(
        "TDLoss",
        OperatorSpec.create(
            name="TDLoss", 
            inputs={"batch": TransitionBatch}, 
            outputs=Tensor((), "float32"),
            allowed_contexts={"learner"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        ),
    )


def test_schema_missing_done_fails() -> None:
    """Verifies that missing fields in a schema are caught (E310)."""
    # Create a schema missing the 'done' field
    IncompleteBatch = Schema(
        fields=[
            Field("obs", BatchObs),
            Field("action", Tensor((-1,), "int64")),
            Field("reward", Tensor((-1,), "float32")),
            Field("next_obs", BatchObs),
            # 'done' is intentionally missing
        ]
    )

    register_spec(
        "IncompleteSampler", OperatorSpec.create(
            name="IncompleteSampler", 
            inputs={}, 
            outputs=IncompleteBatch,
            allowed_contexts={"actor"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        )
    )

    g = Graph()
    g.add_node("sampler", "IncompleteSampler")
    g.add_node("loss", "TDLoss")
    g.add_edge("sampler", "loss", dst_port="batch")

    report = validate_ports(g)
    assert report.has_errors()
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(
        i.code == "E310" and "done" in i.message for i in issues
    ), "Expected E310 for missing 'done' field"


def test_schema_wrong_dtype_fails() -> None:
    """Verifies that field dtype mismatches in schemas are caught (E311)."""
    # Create a schema where 'action' is float32 instead of int64
    WrongDtypeBatch = Schema(
        fields=[
            Field("obs", BatchObs),
            Field("action", Tensor((-1,), "float32")),  # Wrong dtype: expected int64
            Field("reward", Tensor((-1,), "float32")),
            Field("next_obs", BatchObs),
            Field("done", Tensor((-1,), "bool")),
        ]
    )

    register_spec(
        "WrongDtypeSampler", OperatorSpec.create(
            name="WrongDtypeSampler", 
            inputs={}, 
            outputs=WrongDtypeBatch,
            allowed_contexts={"actor"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        )
    )

    g = Graph()
    g.add_node("sampler", "WrongDtypeSampler")
    g.add_node("loss", "TDLoss")
    g.add_edge("sampler", "loss", dst_port="batch")

    report = validate_ports(g)
    assert report.has_errors()
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(
        i.code == "E311" and "action" in i.message and "float32" in i.message
        for i in issues
    ), "Expected E311 for 'action' dtype mismatch"


def test_schema_valid_batch_passes() -> None:
    """Verifies that a valid TransitionBatch passes schema validation."""
    register_spec(
        "ValidSampler", OperatorSpec.create(
            name="ValidSampler", 
            inputs={}, 
            outputs=TransitionBatch,
            allowed_contexts={"actor"},
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
        )
    )

    g = Graph()
    g.add_node("sampler", "ValidSampler")
    g.add_node("loss", "TDLoss")
    g.add_edge("sampler", "loss", dst_port="batch")

    report = validate_ports(g)
    assert not report.has_errors(), f"Valid schema failed validation: {report.issues}"
