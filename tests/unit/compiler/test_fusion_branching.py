import pytest
from core.graph import Graph, NODE_TYPE_SINK
from compiler.rewrite import RewriteEngine, FusionRule, find_linear_chain
from runtime.specs import register_spec, OperatorSpec

pytestmark = pytest.mark.unit

def test_fusion_no_fuse_multi_consumer() -> None:
    """Verifies that fusion is blocked if the source node has multiple consumers."""
    register_spec("BranchA", OperatorSpec.create(name="BranchA", pure=True, deterministic=True))
    register_spec("BranchB", OperatorSpec.create(name="BranchB", pure=True, deterministic=True))
    register_spec("BranchC", OperatorSpec.create(name="BranchC", pure=True, deterministic=True))
    
    g = Graph()
    g.add_node("n1", "BranchA")
    g.add_node("n2", "BranchB")
    g.add_node("n3", "BranchB")
    g.add_edge("n1", "n2")
    g.add_edge("n1", "n3") # n1 has two consumers
    
    chain = find_linear_chain(g, ["BranchA", "BranchB"])
    assert chain == [], "Should not match chain because n1 has multiple consumers"

def test_fusion_no_fuse_multi_producer() -> None:
    """Verifies that fusion is blocked if the destination node has multiple producers."""
    g = Graph()
    g.add_node("n1", "BranchA")
    g.add_node("n2", "BranchA")
    g.add_node("n3", "BranchB")
    g.add_edge("n1", "n3")
    g.add_edge("n2", "n3") # n3 has two producers
    
    chain = find_linear_chain(g, ["BranchA", "BranchB"])
    assert chain == [], "Should not match chain because n3 has multiple producers"
