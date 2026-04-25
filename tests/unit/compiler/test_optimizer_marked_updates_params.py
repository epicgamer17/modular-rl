import pytest
from runtime.registry import get_spec
from agents.dqn.specs import register_dqn_specs

pytestmark = pytest.mark.unit


def test_optimizer_marked_updates_params() -> None:
    """
    Optimizer spec must expose update effect.
    """
    # Ensure specs are registered
    register_dqn_specs()
    
    spec = get_spec("Optimizer")
    assert spec is not None, "Optimizer spec not found"
    
    # Check gradient semantics
    assert spec.consumes_grad is True, "Optimizer must consume gradients"
    assert spec.updates_params is True, "Optimizer must mark updates_params=True"
    assert spec.differentiable is True, "Optimizer is part of the differentiable chain"
