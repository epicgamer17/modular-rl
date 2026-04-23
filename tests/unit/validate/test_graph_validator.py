import pytest
from core.graph import Graph, NODE_TYPE_ACTOR, NODE_TYPE_SOURCE, NODE_TYPE_SINK
from core.schema import Schema, Field, TensorSpec, TAG_ON_POLICY, TAG_ORDERED
from validate.graph_validator import validate_graph

pytestmark = pytest.mark.unit

def test_valid_ppo_graph():
    """Valid PPO graph should pass validation."""
    graph = Graph()
    
    # 1. Define schemas
    obs_schema = Schema(fields=[Field("obs", TensorSpec((4,), "float32", tags=[TAG_ORDERED]))])
    action_schema = Schema(fields=[Field("action", TensorSpec((1,), "int64"))])
    
    # 2. Add nodes
    graph.add_node("env", NODE_TYPE_SOURCE, schema_out=obs_schema)
    graph.add_node("ppo_actor", NODE_TYPE_ACTOR, 
                   schema_in=obs_schema, 
                   schema_out=action_schema,
                   tags=["PPO", TAG_ON_POLICY])
    
    # 3. Add edge
    graph.add_edge("env", "ppo_actor")
    
    # 4. Validate
    validate_graph(graph) # Should not raise

def test_ppo_with_replay_fails():
    """PPO should not be allowed to consume data from a Replay buffer (semantic constraint)."""
    graph = Graph()
    
    obs_schema = Schema(fields=[Field("obs", TensorSpec((4,), "float32"))])
    
    # 1. Add replay buffer
    graph.add_node("replay", NODE_TYPE_SINK, schema_out=obs_schema, tags=["Replay"])
    
    # 2. Add PPO actor consuming from replay
    graph.add_node("ppo_actor", NODE_TYPE_ACTOR, 
                   schema_in=obs_schema, 
                   tags=["PPO", TAG_ON_POLICY])
    
    graph.add_edge("replay", "ppo_actor")
    
    # 3. Validate should fail
    with pytest.raises(ValueError, match="cannot consume data from Replay buffer"):
        validate_graph(graph)

def test_mismatched_shapes_fails():
    """Graph with mismatched tensor shapes should fail validation."""
    graph = Graph()
    
    # Source outputs (4,)
    schema_4 = Schema(fields=[Field("data", TensorSpec((4,), "float32"))])
    # Sink expects (8,)
    schema_8 = Schema(fields=[Field("data", TensorSpec((8,), "float32"))])
    
    graph.add_node("src", NODE_TYPE_SOURCE, schema_out=schema_4)
    graph.add_node("dst", NODE_TYPE_SINK, schema_in=schema_8)
    
    graph.add_edge("src", "dst")
    
    # Validate should fail
    with pytest.raises(ValueError, match="Type mismatch"):
        validate_graph(graph)

def test_cycle_detection():
    """Graph with cycles should fail validation."""
    graph = Graph()
    
    graph.add_node("a", "Type")
    graph.add_node("b", "Type")
    
    graph.add_edge("a", "b")
    graph.add_edge("b", "a")
    
    with pytest.raises(ValueError, match="Cycle detected"):
        validate_graph(graph)

if __name__ == "__main__":
    test_valid_ppo_graph()
    try:
        test_ppo_with_replay_fails()
    except ValueError as e:
        print(f"Caught expected PPO/Replay error: {e}")
    
    try:
        test_mismatched_shapes_fails()
    except ValueError as e:
        print(f"Caught expected Shape mismatch error: {e}")
        
    try:
        test_cycle_detection()
    except ValueError as e:
        print(f"Caught expected Cycle error: {e}")
        
    print("Test 2.1 Passed!")
