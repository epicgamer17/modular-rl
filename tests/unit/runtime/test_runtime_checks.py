import pytest
import torch
from core.graph import Graph
from runtime.executor import execute, register_operator
from runtime.values import NoOp

pytestmark = pytest.mark.unit


def test_runtime_none_forbidden() -> None:
    """Verifies that an operator returning raw None raises a RuntimeError."""
    g = Graph()
    # Register a bad operator that explicitly returns None
    register_operator("BadNode", lambda node, inputs, context: None)
    g.add_node("bad", "BadNode")

    with pytest.raises(RuntimeError, match="returned raw None"):
        execute(g, initial_inputs={})


def test_runtime_nan_forbidden() -> None:
    """Verifies that an operator returning NaN raises a RuntimeError."""
    g = Graph()
    # Register an operator that returns NaN
    register_operator(
        "NanNode", lambda node, inputs, context: torch.tensor(float("nan"))
    )
    g.add_node("nan", "NanNode")

    with pytest.raises(RuntimeError, match="produced non-finite values"):
        execute(g, initial_inputs={})


def test_runtime_loss_scalar_enforcement() -> None:
    """Verifies that a node with 'loss' in its type must produce a scalar."""
    g = Graph()
    # Node type contains 'Loss', triggering scalar enforcement
    register_operator("QLoss", lambda node, inputs, context: torch.ones(2, 2))
    g.add_node("loss_node", "QLoss")

    with pytest.raises(RuntimeError, match="produced non-scalar output"):
        execute(g, initial_inputs={})


def test_runtime_negative_action_forbidden() -> None:
    """Verifies that discrete actions (integers) must be non-negative."""
    g = Graph()
    # Outputting a dictionary with an 'action' field that is negative
    register_operator(
        "BadActor", lambda node, inputs, context: {"action": torch.tensor(-1)}
    )
    g.add_node("actor", "BadActor")

    with pytest.raises(RuntimeError, match="produced negative discrete action"):
        execute(g, initial_inputs={})


def test_runtime_valid_outputs_pass() -> None:
    """Verifies that valid, finite outputs pass validation without error."""
    g = Graph()
    register_operator("ValidNode", lambda node, inputs, context: torch.tensor(1.0))
    g.add_node("valid", "ValidNode")

    results = execute(g, initial_inputs={})
    # Results are wrapped in Value objects by the executor
    assert results["valid"].data == 1.0


def test_runtime_noop_passes() -> None:
    """Verifies that NoOp (explicit null-operation) passes validation."""
    g = Graph()
    register_operator("NoOpNode", lambda node, inputs, context: NoOp())
    g.add_node("noop", "NoOpNode")

    results = execute(g, initial_inputs={})
    assert isinstance(results["noop"], NoOp)
