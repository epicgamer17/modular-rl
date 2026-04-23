import pytest
from core.graph import Graph, NODE_TYPE_SOURCE, NODE_TYPE_ACTOR, NODE_TYPE_SINK
from core.schema import TAG_ON_POLICY, TAG_OFF_POLICY
from compiler.analyzer import analyze_graph

pytestmark = pytest.mark.unit

def test_detect_unused_nodes():
    """Verify that unused sources and dangling nodes are detected as warnings."""
    graph = Graph()
    graph.add_node("obs", NODE_TYPE_SOURCE)
    graph.add_node("dangling", "Actor")
    
    analysis = analyze_graph(graph)
    
    assert len(analysis.warnings) == 2
    assert any("Unused Source" in w for w in analysis.warnings)
    assert any("Dangling node" in w for w in analysis.warnings)

def test_detect_disconnected_nodes():
    """Verify that disconnected non-source nodes are detected as errors."""
    graph = Graph()
    graph.add_node("middle", "Transform") # No incoming edges
    
    analysis = analyze_graph(graph)
    
    assert not analysis.is_valid()
    assert any("Disconnected node" in e for e in analysis.errors)

def test_detect_invalid_ppo_semantics():
    """Verify that PPO nodes missing OnPolicy tags are detected."""
    graph = Graph()
    graph.add_node("ppo_loss", "PPOObjective", tags=[]) # Should have TAG_ON_POLICY
    
    analysis = analyze_graph(graph)
    
    assert not analysis.is_valid()
    assert any("PPO Violation" in e for e in analysis.errors)

def test_detect_semantic_conflicts():
    """Verify that conflicting tags (e.g. Replay + OnPolicy) are detected."""
    graph = Graph()
    graph.add_node("replay", "ReplayAdd", tags=[TAG_ON_POLICY]) # Replay is OffPolicy
    
    analysis = analyze_graph(graph)
    
    assert not analysis.is_valid()
    assert any("Semantic Conflict" in e for e in analysis.errors)

if __name__ == "__main__":
    test_detect_unused_nodes()
    test_detect_disconnected_nodes()
    test_detect_invalid_ppo_semantics()
    test_detect_semantic_conflicts()
    print("Graph Analyzer Tests Passed!")
