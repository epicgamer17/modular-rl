import pytest
from core.graph import Graph
from observability.graph_viz.render import render_graph
import os

pytestmark = pytest.mark.unit

def test_render_static_graph():
    graph = Graph()
    graph.add_node("obs", "Source")
    graph.add_node("policy", "Actor")
    graph.add_node("act", "Sink")
    graph.add_edge("obs", "policy")
    graph.add_edge("policy", "act")
    
    # Test rendering to file (to avoid GUI dependency in tests)
    save_path = "test_graph.png"
    render_graph(graph, mode="static", save_path=save_path)
    
    assert os.path.exists(save_path)
    os.remove(save_path)

def test_render_compiler_view():
    graph = Graph()
    n1 = graph.add_node("n1", "Transform", params={"compiler_state": "fused"})
    n2 = graph.add_node("n2", "Transform", params={"compiler_state": "pruned"})
    graph.add_edge("n1", "n2")
    
    save_path = "test_compiler_view.png"
    render_graph(graph, mode="compiler", save_path=save_path)
    
    assert os.path.exists(save_path)
    os.remove(save_path)
