import pytest
from core.graph import Graph
from agents.dqn.specs import register_dqn_specs
from compiler.passes.shape.gradient_analysis import analyze_gradients

pytestmark = pytest.mark.unit

def test_detect_dead_parameter():
    """Verifies that models with no loss path are flagged as dead."""
    register_dqn_specs()
    g = Graph()
    
    # Valid path: online_q -> QValues -> TDLoss -> Optimizer
    g.add_node("q_node", "QValuesSingle", params={"model_handle": "online_q"})
    g.add_node("loss", "TDLoss")
    g.add_node("opt", "Optimizer")
    g.add_edge("q_node", "loss")
    g.add_edge("loss", "opt")
    
    # Dead path: dead_model -> QValues -> (Nothing)
    g.add_node("dead_node", "QValuesSingle", params={"model_handle": "dead_model"})
    
    report = analyze_gradients(g)
    
    assert "online_q" in report.params_with_grad
    assert "dead_model" in report.params_without_grad
    assert any("Dead parameter detected: 'dead_model'" in w for w in report.warnings)

def test_detect_unused_branch():
    """Verifies that differentiable nodes with no consumers are flagged."""
    register_dqn_specs()
    g = Graph()
    g.add_node("q_node", "QValuesSingle", params={"model_handle": "online_q"})
    # q_node is differentiable but has no successor
    
    report = analyze_gradients(g)
    assert "q_node" in report.unused_branches
    assert any("Unused differentiable branch: 'q_node'" in w for w in report.warnings)
