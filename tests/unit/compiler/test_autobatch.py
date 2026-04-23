import pytest
from core.graph import Graph, NODE_TYPE_SOURCE
from core.schema import Schema, Field, TensorSpec
from core.types import TensorType
from compiler.passes.autobatch import vectorize_graph

pytestmark = pytest.mark.unit

def test_vectorize_single_node():
    """Verify that a single node's schema is correctly up-ranked with a batch dimension."""
    # Define a single-step observation type [D]
    obs_type = TensorType(shape=(4,), dtype="float32", tags={"obs"})
    obs_schema = Schema(fields=[Field("obs", TensorSpec(shape=(4,), dtype="float32", rl_type=obs_type))])
    
    # Define a single-step action type [A]
    action_type = TensorType(shape=(2,), dtype="float32", tags={"action"})
    action_schema = Schema(fields=[Field("action", TensorSpec(shape=(2,), dtype="float32", rl_type=action_type))])
    
    graph = Graph()
    graph.add_node("policy", "PolicyActor", schema_in=obs_schema, schema_out=action_schema)
    
    vectorized_graph = vectorize_graph(graph)
    
    node = vectorized_graph.nodes["policy"]
    
    # Check input schema
    obs_field = node.schema_in.fields[0]
    assert obs_field.spec.shape == (-1, 4)
    assert "batched" in obs_field.spec.tags
    assert obs_field.spec.rl_type.shape == ("B", 4)
    assert "batched" in obs_field.spec.rl_type.tags
    
    # Check output schema
    action_field = node.schema_out.fields[0]
    assert action_field.spec.shape == (-1, 2)
    assert "batched" in action_field.spec.tags
    assert action_field.spec.rl_type.shape == ("B", 2)
    assert "batched" in action_field.spec.rl_type.tags
    
    assert "vectorized" in node.tags

def test_vectorize_propagation():
    """Verify that vectorization propagates through edges."""
    obs_schema = Schema(fields=[Field("obs", TensorSpec(shape=(4,), dtype="float32"))])
    
    graph = Graph()
    graph.add_node("src", NODE_TYPE_SOURCE, schema_out=obs_schema)
    graph.add_node("dst", "Actor", schema_in=obs_schema)
    graph.add_edge("src", "dst")
    
    vectorized_graph = vectorize_graph(graph)
    
    src = vectorized_graph.nodes["src"]
    dst = vectorized_graph.nodes["dst"]
    
    assert src.schema_out.fields[0].spec.shape == (-1, 4)
    assert dst.schema_in.fields[0].spec.shape == (-1, 4)
    assert len(vectorized_graph.edges) == 1

def test_no_batch_tag():
    """Verify that nodes with 'no_batch' tag are not vectorized."""
    obs_schema = Schema(fields=[Field("obs", TensorSpec(shape=(4,), dtype="float32"))])
    
    graph = Graph()
    graph.add_node("sink", "Sink", schema_in=obs_schema, tags=["no_batch"])
    
    vectorized_graph = vectorize_graph(graph)
    
    node = vectorized_graph.nodes["sink"]
    assert node.schema_in.fields[0].spec.shape == (4,)
    assert "vectorized" not in node.tags
