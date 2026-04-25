import pytest
from runtime.registry import OperatorSpec

pytestmark = pytest.mark.unit


def test_learner_op_requires_trainability_metadata() -> None:
    """
    Every learner-side op must explicitly declare gradient semantics.
    """
    # 1. Missing all metadata should fail
    with pytest.raises(ValueError, match="must explicitly declare: differentiable, creates_grad, consumes_grad, updates_params"):
        OperatorSpec.create(
            name="MissingMetadataOp",
            allowed_contexts={"learner"}
        )

    # 2. Missing some metadata should fail
    with pytest.raises(ValueError, match="must explicitly declare: creates_grad, consumes_grad, updates_params"):
        OperatorSpec.create(
            name="PartialMetadataOp",
            allowed_contexts={"learner"},
            differentiable=True
        )

    # 3. Explicitly providing all should pass
    spec = OperatorSpec.create(
        name="ValidMetadataOp",
        allowed_contexts={"learner"},
        differentiable=True,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False
    )
    assert spec.differentiable is True
    assert spec.creates_grad is False


def test_actor_op_does_not_require_metadata() -> None:
    """
    Actor-only ops should still use defaults and not raise errors.
    """
    spec = OperatorSpec.create(
        name="ActorOnlyOp",
        allowed_contexts={"actor"}
    )
    assert spec.differentiable is True # default
    assert spec.creates_grad is False # default
