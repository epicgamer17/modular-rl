import pytest
from core.graph import Graph, NODE_TYPE_SINK
from compiler.rewrite import RewriteEngine, FusionRule
from runtime.registry import register_spec, OperatorSpec

pytestmark = pytest.mark.unit

def test_fusion_simple_chain() -> None:
    """Verifies that a basic A -> B chain is fused into C."""
    register_spec("BasicA", OperatorSpec.create(name="BasicA", pure=True, deterministic=True))
    register_spec("BasicB", OperatorSpec.create(name="BasicB", pure=True, deterministic=True))
    register_spec("BasicC", OperatorSpec.create(name="BasicC", pure=True, deterministic=True))
    register_spec(NODE_TYPE_SINK, OperatorSpec.create(name=NODE_TYPE_SINK))
    
    engine = RewriteEngine()
    engine.add_rule(FusionRule(name="a_b_fusion", pattern=["BasicA", "BasicB"], replacement="BasicC"))
    
    g = Graph()
    g.add_node("n1", "BasicA")
    g.add_node("n2", "BasicB")
    g.add_node("sink", NODE_TYPE_SINK)
    g.add_edge("n1", "n2")
    g.add_edge("n2", "sink")
    
    optimized_g = engine.apply(g)
    
    # n1 and n2 should be replaced by fused node
    assert "n1" not in optimized_g.nodes
    assert "n2" not in optimized_g.nodes
    
    fused_nodes = [nid for nid, n in optimized_g.nodes.items() if n.node_type == "BasicC"]
    assert len(fused_nodes) == 1
    
    # Verify connectivity
    assert "sink" in optimized_g.nodes
    # Fused node should now point to sink
    found_edge = False
    for edge in optimized_g.edges:
        if edge.src == fused_nodes[0] and edge.dst == "sink":
            found_edge = True
            break
    assert found_edge, "Fused node should be connected to the sink"
