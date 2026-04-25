import pytest
from core.graph import Graph, Edge
from compiler.rewrite import rewrite, FusionRule
from runtime.registry import register_spec, OperatorSpec
from core.schema import TensorSpec

pytestmark = pytest.mark.unit

def test_fusion_ports_preserved() -> None:
    """Verifies that ports (dst_port) are correctly re-mapped during fusion."""
    register_spec("PortInput", OperatorSpec.create(name="PortInput", pure=True, deterministic=True))
    register_spec("PortOpA", OperatorSpec.create(name="PortOpA", pure=True, deterministic=True))
    register_spec("PortOpB", OperatorSpec.create(name="PortOpB", pure=True, deterministic=True))
    register_spec("PortFused", OperatorSpec.create(name="PortFused", pure=True, deterministic=True))
    
    g = Graph()
    g.add_node("src", "PortInput")
    g.add_node("a", "PortOpA")
    g.add_node("b", "PortOpB")
    
    # Connect src to 'a' with a specific port
    g.add_edge("src", "a", dst_port="target_port")
    # Linear connection between 'a' and 'b'
    g.add_edge("a", "b")
    
    rule = FusionRule(name="test_fusion", pattern=["PortOpA", "PortOpB"], replacement="PortFused")
    
    optimized_g = rewrite(g, ["a", "b"], rule)
    
    # Fused node ID is deterministic based on head and tail
    fused_id = "a_b_fused"
    assert fused_id in optimized_g.nodes
    
    # Check incoming edge port
    incoming_edges = [e for e in optimized_g.edges if e.dst == fused_id]
    assert len(incoming_edges) == 1
    assert incoming_edges[0].dst_port == "target_port", "Incoming port name should be preserved"
