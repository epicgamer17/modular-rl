import pytest
from core.graph import Graph
from runtime.specs import register_spec, OperatorSpec
from compiler.compiler import compile_graph
from compiler.validation import SEVERITY_WARN, SEVERITY_ERROR
from compiler.passes.validate_purity import validate_purity

pytestmark = pytest.mark.unit


def test_impure_node_duplicated_warning() -> None:
    """Verifies that duplicated side effects trigger D001 warning."""
    register_spec(
        "SideEffectOp",
        OperatorSpec.create(name="SideEffectOp", side_effects=["update_weights"]),
    )

    g = Graph()
    g.add_node("n1", "SideEffectOp")
    g.add_node("n2", "SideEffectOp")

    report = validate_purity(g)

    assert report.has_warnings()
    issues = report.get_issues_by_severity(SEVERITY_WARN)
    assert any(i.code == "D001" and "update_weights" in i.message for i in issues)


def test_optimizer_in_actor_graph_error() -> None:
    """Verifies that optimizer-requiring nodes in actor context trigger D003."""
    register_spec(
        "OptimizerOp", OperatorSpec.create(name="OptimizerOp", requires_optimizer=True)
    )

    g = Graph()
    g.add_node("opt", "OptimizerOp")

    # This should fail with D003 in actor context
    with pytest.raises(RuntimeError, match="D003"):
        compile_graph(g, context="actor")

    # Should pass in learner context (default allows 'both' if not specified, 
    # but requires_optimizer is allowed in learner)
    compile_graph(g, context="learner")


def test_pure_node_reused_allowed() -> None:
    """Verifies that pure nodes can be duplicated without warnings."""
    register_spec("PureOp", OperatorSpec.create(name="PureOp", pure=True))

    g = Graph()
    g.add_node("p1", "PureOp")
    g.add_node("p2", "PureOp")

    report = validate_purity(g)
    assert not report.has_warnings()


def test_context_violation_error() -> None:
    """Verifies that nodes not allowed in context trigger D002."""
    register_spec(
        "LearnerOnlyOp",
        OperatorSpec.create(name="LearnerOnlyOp", allowed_contexts={"learner"}),
    )

    g = Graph()
    g.add_node("l", "LearnerOnlyOp")

    # Should fail in actor context
    with pytest.raises(RuntimeError, match="D002"):
        compile_graph(g, context="actor")

    # Should pass in learner context
    compile_graph(g, context="learner")
