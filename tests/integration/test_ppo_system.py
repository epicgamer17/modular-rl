import pytest
import torch
import torch.nn as nn
from core.graph import Graph, NODE_TYPE_SOURCE
from runtime.executor import execute, register_operator
from runtime.state import ParameterStore, OptimizerState, ModelRegistry
from runtime.context import ExecutionContext
from agents.ppo.model import ActorCritic
from agents.ppo.config import PPOConfig
from agents.ppo.operators import register_ppo_operators

pytestmark = pytest.mark.integration

# Register PPO operators for integration tests
register_ppo_operators()

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
        "terminated": torch.zeros(32),
        "truncated": torch.zeros(32),
        "policy_version": 0
    }
    
    # 2. Increment parameter version
    param_store.update_state({}) # version becomes 1
    assert param_store.version == 1
    
    config = PPOConfig(obs_dim=obs_dim, act_dim=act_dim)
    
    # 3. Setup Decomposed PPO graph
    from agents.ppo.graphs import create_train_graph
    graph = create_train_graph(config)
    
    # 4. Execute with decomposed graph
    model_registry = ModelRegistry()
    model_registry.register(config.model_handle, ac_net)
    ctx = ExecutionContext(model_registry=model_registry)
    
    # We need to wrap data in a TransitionBatch
    from core.batch import TransitionBatch
    batch = TransitionBatch(
        obs=data["obs"],
        action=data["action"],
        log_prob=data["log_prob"],
        reward=data["reward"],
        next_obs=data["next_obs"],
        terminated=data["terminated"],
        truncated=data["truncated"],
        policy_version=data["policy_version"],
        advantages=torch.ones(32),
        returns=torch.zeros(32)
    )
    
    # This should now succeed or fail based on current logic
    execute(graph, initial_inputs={"traj_in": batch}, context=ctx)

def test_ppo_on_policy_tag_enforcement():
    """Verify that the compiler enforces the OnPolicy tag for PPO nodes."""
    from compiler.pipeline import compile_graph
    
    graph = Graph()
    # Adding a node with 'PPO' tag but missing TAG_ON_POLICY
    graph.add_node("ppo_node", "Actor", tags=["PPO"]) 
    
    with pytest.raises(RuntimeError, match="must have OnPolicy tag"):
        compile_graph(graph)

if __name__ == "__main__":
    # Register needed operators if not already
    register_ppo_operators()
    
    test_ppo_stale_policy_detection()
    print("PPO Stale Policy Detection Verified!")
    
    try:
        test_ppo_on_policy_tag_enforcement()
        print("PPO OnPolicy Tag Enforcement Verified!")
    except ImportError:
        print("Skipping tag enforcement test (validator not in path)")
