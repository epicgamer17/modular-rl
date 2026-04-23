import pytest
from core.graph import Graph, NodeId, EdgeType

pytestmark = pytest.mark.unit

def test_minimal_graph_construction():
    """
    Test 1.1: Verify minimal graph construction and adjacency lists.
    - create empty graph
    - add nodes
    - connect nodes
    - assert adjacency lists correct
    """
    # 1. create empty graph
    graph = Graph()
    
    # 2. add nodes
    graph.add_node("actor", "Actor", tags=["pi_network"])
    graph.add_node("obs", "Source", tags=["environment"])
    graph.add_node("replay", "Sink", tags=["memory"])
    
    # 3. connect nodes
    graph.add_edge("obs", "actor", edge_type=EdgeType.DATA)
    graph.add_edge("actor", "replay", edge_type=EdgeType.DATA)
    
    # 4. assert adjacency lists correct
    adj = graph.adjacency_list
    
    assert NodeId("obs") in adj
    assert NodeId("actor") in adj["obs"]
    
    assert NodeId("actor") in adj
    assert NodeId("replay") in adj["actor"]
    
    assert NodeId("replay") in adj
    assert len(adj["replay"]) == 0
    
    print("\nAdjacency List:")
    print(adj)

if __name__ == "__main__":
    # Allow running this test directly to see output
    test_minimal_graph_construction()
    print("Test 1.1 Passed!")
