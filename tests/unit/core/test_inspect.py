import pytest
from core.graph import Graph, NODE_TYPE_ACTOR, NODE_TYPE_SOURCE
from core.schema import Schema, Field, TensorSpec, TAG_ON_POLICY
from core.inspect import print_graph_summary, trace_node_lineage, display_schema_propagation

pytestmark = pytest.mark.unit

def test_ppo_graph_introspection(capsys):
    """Verify that PPO graph can be summarized and traced."""
    graph = Graph()
    
    obs_schema = Schema(fields=[Field("obs", TensorSpec((4,), "float32"))])
    action_schema = Schema(fields=[Field("action", TensorSpec((1,), "int64"))])
    
    graph.add_node("env", NODE_TYPE_SOURCE, schema_out=obs_schema)
    graph.add_node("ppo_actor", NODE_TYPE_ACTOR, 
                   schema_in=obs_schema, 
                   schema_out=action_schema,
                   tags=["PPO", TAG_ON_POLICY])
    
    graph.add_edge("env", "ppo_actor")
    
    # 1. Print summary
    print_graph_summary(graph)
    captured = capsys.readouterr()
    assert "Graph Summary" in captured.out
    assert "ppo_actor" in captured.out
    assert "env" in captured.out
    
    # 2. Trace lineage
    lineage = trace_node_lineage(graph, "ppo_actor")
    assert "env" in lineage["upstream"]
    assert len(lineage["downstream"]) == 0
    
    # 3. Display schema propagation
    display_schema_propagation(graph)
    captured = capsys.readouterr()
    assert "Schema Propagation" in captured.out
    assert "Compatible: YES" in captured.out

if __name__ == "__main__":
    # For manual verification
    from core.graph import Graph, NODE_TYPE_ACTOR, NODE_TYPE_SOURCE
    from core.schema import Schema, Field, TensorSpec, TAG_ON_POLICY
    from core.inspect import print_graph_summary, trace_node_lineage, display_schema_propagation

    graph = Graph()
    obs_schema = Schema(fields=[Field("obs", TensorSpec((4,), "float32"))])
    action_schema = Schema(fields=[Field("action", TensorSpec((1,), "int64"))])
    graph.add_node("env", NODE_TYPE_SOURCE, schema_out=obs_schema)
    graph.add_node("ppo_actor", NODE_TYPE_ACTOR, schema_in=obs_schema, schema_out=action_schema, tags=["PPO", TAG_ON_POLICY])
    graph.add_edge("env", "ppo_actor")
    
    print_graph_summary(graph)
    print("\nLineage of ppo_actor:", trace_node_lineage(graph, "ppo_actor"))
    display_schema_propagation(graph)
    print("Test 2.2 Passed!")
