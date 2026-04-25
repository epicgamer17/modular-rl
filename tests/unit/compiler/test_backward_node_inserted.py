import pytest
import torch
from core.graph import Graph
from agents.dqn.specs import register_dqn_specs
from runtime.registry import register_base_specs, clear_registry
from compiler.pipeline import compile_graph

pytestmark = pytest.mark.unit

@pytest.fixture(autouse=True)
def setup_specs():
    clear_registry()
    register_base_specs()
    register_dqn_specs()

def test_backward_node_inserted():
    """Verifies that the autodiff pass inserts Backward and GradBuffer nodes."""
    g = Graph()
    
    # 1. Forward/Loss
    g.add_node("pred", "Source")
    g.add_node("target", "Source")
    g.add_node("mse", "MSELoss")
    g.add_edge("pred", "mse", dst_port="pred")
    g.add_edge("target", "mse", dst_port="target")
    
    # 2. Optimizer
    g.add_node("opt", "Optimizer", params={"model_handle": "online_q"})
    g.add_edge("mse", "opt", dst_port="loss")
    
    # Compile with autodiff lowering
    compiled = compile_graph(g, context="learner", model_handles={"online_q"}, autodiff_lowering=True)
    
    # Verify nodes
    node_types = [n.node_type for n in compiled.nodes.values()]
    assert "Backward" in node_types
    assert "GradBuffer" in node_types
    
    # Verify connectivity
    # Backward should be connected to MSELoss
    backward_node = next(nid for nid, n in compiled.nodes.items() if n.node_type == "Backward")
    edges_to_backward = [e for e in compiled.edges if e.dst == backward_node]
    assert any(e.src == "mse" for e in edges_to_backward)
