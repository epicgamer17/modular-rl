import pytest
import warnings
from core.graph import Graph
from runtime.specs import OperatorSpec, register_spec, get_spec, TransitionBatch, ScalarLoss
from compiler.compiler import compile_graph

pytestmark = pytest.mark.unit


def test_operator_metadata_registration() -> None:
    """Verifies that an operator can be registered with full metadata."""
    spec = OperatorSpec.create(
        name="AdvancedLearner",
        version="2.1.0",
        inputs={"batch": TransitionBatch},
        outputs={"loss": ScalarLoss},
        pure=False,
        stateful=True,
        deterministic=False,
        requires_models=["main"],
        requires_optimizer=True,
        allowed_contexts={"learner"},
        differentiable=True,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
        tags={"off_policy", "heavy"},
    )

    register_spec("AdvancedLearner", spec)
    retrieved = get_spec("AdvancedLearner")

    assert retrieved is not None
    assert retrieved.name == "AdvancedLearner"
    assert retrieved.version == "2.1.0"
    assert retrieved.pure is False
    assert retrieved.stateful is True
    assert retrieved.requires_optimizer is True
    assert "learner" in retrieved.allowed_contexts
    assert "off_policy" in retrieved.tags


def test_operator_metadata_defaults() -> None:
    """Verifies that defaults are correctly applied when metadata is missing."""
    spec = OperatorSpec.create(name="SimpleOp", allowed_contexts={"actor"})

    assert spec.version == "1.0.0"
    assert spec.pure is False
    assert spec.requires_optimizer is False
    assert spec.allowed_contexts == {"actor"}
    assert spec.tags == set()


def test_duplicate_version_warning() -> None:
    """Verifies that registering the same version with different specs issues a warning."""
    spec1 = OperatorSpec.create(name="VersionedOp", version="1.0.0", pure=True, allowed_contexts={"actor"})
    spec2 = OperatorSpec.create(name="VersionedOp", version="1.0.0", pure=False, allowed_contexts={"actor"})

    register_spec("VersionedOp", spec1)
    with pytest.warns(UserWarning, match="Duplicate version '1.0.0'"):
        register_spec("VersionedOp", spec2)


def test_missing_metadata_fails_strict_mode() -> None:
    """Verifies that missing metadata triggers an error in strict mode (M001)."""
    g = Graph()
    g.add_node("unknown", "UnregisteredType")

    # In normal mode, it should be a warning (which might still fail compile_graph if strict=False but we check for warnings)
    # Actually, compile_graph with strict=False will NOT raise for M001 (warning)
    compile_graph(g, strict=False)

    # In strict mode, it should raise RuntimeError due to M001 being an ERROR
    with pytest.raises(RuntimeError, match="M001"):
        compile_graph(g, strict=True)
