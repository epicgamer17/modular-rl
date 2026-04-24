import pytest
from core.graph import Graph
from agents.dqn.specs import register_dqn_specs
from runtime.specs import register_spec, OperatorSpec
from compiler.passes.analyze_gradients import analyze_gradients

pytestmark = pytest.mark.unit

def test_detect_detach_break():
    """Verifies that non-differentiable operators in a gradient path trigger warnings."""
    register_dqn_specs()
    register_spec(
        "StopGrad",
        OperatorSpec.create(
            name="StopGrad",
            pure=True,
            differentiable=False,
            creates_grad=False,
            consumes_grad=False,
            updates_params=False,
            allowed_contexts={"learner"}
        )
    )
    
    g = Graph()
    g.add_node("q_node", "QValuesSingle", params={"model_handle": "online_q"})
    g.add_node("detach", "StopGrad")
    g.add_node("loss", "TDLoss")
    g.add_node("opt", "Optimizer")
    
    g.add_edge("q_node", "detach")
    g.add_edge("detach", "loss")
    g.add_edge("loss", "opt")
    
    report = analyze_gradients(g)
    
    # The gradient path is broken, so online_q should be "without grad" from the optimizer's perspective
    assert "online_q" in report.params_without_grad
    assert any("Detached gradient flow at edge q_node -> detach" in w for w in report.warnings)
    assert any("StopGrad" in w for w in report.warnings)
