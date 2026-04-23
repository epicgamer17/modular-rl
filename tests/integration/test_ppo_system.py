import pytest
import torch
import torch.nn as nn
from core.graph import Graph, NODE_TYPE_SOURCE
from runtime.executor import execute, register_operator
from runtime.state import ParameterStore, OptimizerState, ModelRegistry
from runtime.context import ExecutionContext
from examples.ppo import ActorCritic, op_gae, op_ppo_objective

pytestmark = pytest.mark.integration

def test_ppo_stale_policy_detection():
    """Verify that PPOObjective detects and rejects stale policy data."""
    obs_dim = 4
    act_dim = 2
    ac_net = ActorCritic(obs_dim, act_dim)
    param_store = ParameterStore(dict(ac_net.named_parameters()))
    
    # 1. Generate data with version 0
    # Mock PolicyActor output
    data = {
        "obs": torch.randn(32, obs_dim),
        "action": torch.zeros(32),
        "log_prob": torch.zeros(32),
        "reward": torch.zeros(32),
        "next_obs": torch.randn(32, obs_dim),
        "done": torch.zeros(32),
        "policy_version": 0
    }
    
    # 2. Increment parameter version
    param_store.update_state({}) # version becomes 1
    assert param_store.version == 1
    
    # 3. Setup PPO graph
    graph = Graph()
    graph.add_node("traj_in", NODE_TYPE_SOURCE)
    graph.add_node("gae", "GAE", params={
        "model_handle": "ppo_net", 
        "gamma": 0.99, 
        "gae_lambda": 0.95
    })
    graph.add_node("ppo", "PPOObjective", params={
        "model_handle": "ppo_net", 
        "param_store_handle": "main_store", 
        "clip_epsilon": 0.2, 
        "strict_on_policy": True
    })
    
    graph.add_edge("traj_in", "gae", dst_port="batch")
    graph.add_edge("traj_in", "ppo", dst_port="batch")
    graph.add_edge("gae", "ppo", dst_port="gae")
    
    # 4. Execute should fail because data is version 0 but policy is version 1
    model_registry = ModelRegistry()
    model_registry.register("ppo_net", ac_net)
    ctx = ExecutionContext(model_registry=model_registry)
    # Temporary workaround: inject param_store into context
    # In a full system, we'd have a ParameterStoreRegistry
    ctx.param_stores = {"main_store": param_store}
    
    with pytest.raises(ValueError, match="STALE POLICY DETECTED"):
        execute(graph, initial_inputs={"traj_in": data}, context=ctx)

def test_ppo_on_policy_tag_enforcement():
    """Verify that the validator enforces the OnPolicy tag for PPO nodes."""
    from validate.graph_validator import validate_graph
    
    graph = Graph()
    # Adding a node with 'PPO' tag but missing TAG_ON_POLICY
    graph.add_node("ppo_node", "Actor", tags=["PPO"]) 
    
    with pytest.raises(ValueError, match="must have OnPolicy tag"):
        validate_graph(graph)

if __name__ == "__main__":
    # Register needed operators if not already
    register_operator("GAE", op_gae)
    register_operator("PPOObjective", op_ppo_objective)
    
    test_ppo_stale_policy_detection()
    print("PPO Stale Policy Detection Verified!")
    
    try:
        test_ppo_on_policy_tag_enforcement()
        print("PPO OnPolicy Tag Enforcement Verified!")
    except ImportError:
        print("Skipping tag enforcement test (validator not in path)")
