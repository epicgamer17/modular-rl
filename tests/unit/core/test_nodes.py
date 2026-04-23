import pytest
from core.nodes import NodeInstance, registry, create_policy_actor_def, create_gae_def
from core.schema import Schema, Field, TensorSpec
from core.graph import NodeId, NODE_TYPE_ACTOR, NODE_TYPE_TRANSFORM

pytestmark = pytest.mark.unit

def test_node_def_registry_and_instantiation():
    """
    Test 1.3: Verify NodeDef creation, registration, and instance propagation.
    """
    # 1. Define schemas
    obs_schema = Schema(fields=[Field("obs", TensorSpec((4,), "float32"))])
    action_schema = Schema(fields=[Field("action", TensorSpec((1,), "int64"))])
    
    # 2. Create definitions
    actor_def = create_policy_actor_def(obs_schema, action_schema)
    gae_def = create_gae_def(obs_schema, obs_schema) # Using obs_schema as placeholder for GAE
    
    # 3. Register definitions
    registry.register("Actor", actor_def)
    registry.register("Transform", gae_def)
    
    # 4. Retrieve and instantiate
    actor_instance = NodeInstance(
        node_id=NodeId("pi_0"),
        node_def=registry.get("Actor"),
        params={"learning_rate": 1e-3}
    )
    
    # 5. Assert propagation
    assert actor_instance.node_id == "pi_0"
    assert actor_instance.node_type == NODE_TYPE_ACTOR
    assert actor_instance.input_schema == obs_schema
    assert actor_instance.output_schema == action_schema
    assert actor_instance.params["learning_rate"] == 1e-3
    
    # 6. Verify another instance
    gae_instance = NodeInstance(
        node_id=NodeId("gae_0"),
        node_def=registry.get("Transform")
    )
    assert gae_instance.node_type == NODE_TYPE_TRANSFORM
    assert gae_instance.input_schema == obs_schema

if __name__ == "__main__":
    test_node_def_registry_and_instantiation()
    print("Test 1.3 Passed!")
