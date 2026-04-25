import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytest
import copy

from agents.ppo.config import PPOConfig
from agents.ppo.model import ActorCritic
from agents.ppo.operators import register_ppo_operators
from ops.rl.advantage import op_gae as op_ppo_gae
from agents.ppo.graphs import create_train_graph, create_ppo_update_graph
from runtime.context import ExecutionContext
from runtime.state import ModelRegistry, OptimizerState, OptimizerRegistry, BufferRegistry
from runtime.executor import execute
from core.graph import Node, NodeId

from ops.registry import register_all_operators
register_all_operators()

pytestmark = pytest.mark.unit

def test_loop_equivalence():
    """
    Compare the results of a manual Python update loop vs the graph-based LoopNode.
    Ensures that loss and gradient updates are identical.
    """
    # 1. Setup deterministic environment
    torch.manual_seed(42)
    np.random.seed(42)
    
    config = PPOConfig(
        obs_dim=4,
        act_dim=2,
        hidden_dim=8,
        epochs=2,
        minibatch_size=4,
        rollout_steps=8,
        num_envs=1,
        learning_rate=1e-3
    )
    
    register_ppo_operators()
    
    def create_setup():
        torch.manual_seed(42)
        model = ActorCritic(config.obs_dim, config.act_dim, config.hidden_dim)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        model_registry = ModelRegistry()
        model_registry.register(config.model_handle, model)
        
        opt_state = OptimizerState(optimizer)
        optimizer_registry = OptimizerRegistry()
        optimizer_registry.register(config.optimizer_handle, opt_state)
        
        ctx = ExecutionContext(
            model_registry=model_registry,
            optimizer_registry=optimizer_registry
        )
        return model, optimizer, ctx

    # 2. Create dummy data
    obs = torch.randn(config.rollout_steps, config.obs_dim)
    action = torch.randint(0, config.act_dim, (config.rollout_steps,))
    reward = torch.randn(config.rollout_steps)
    terminated = torch.zeros(config.rollout_steps, dtype=torch.bool)
    log_prob = torch.randn(config.rollout_steps)
    values = torch.randn(config.rollout_steps)
    
    from core.batch import TransitionBatch
    batch = TransitionBatch(
        obs=obs,
        action=action,
        reward=reward,
        terminated=terminated,
        truncated=torch.zeros_like(terminated),
        next_obs=torch.randn_like(obs),
        done=terminated | torch.zeros_like(terminated),
        log_prob=log_prob,
        value=values
    )
    
    next_value = torch.randn(1)
    next_terminated = torch.zeros(1, dtype=torch.bool)
    
    # --- APPROACH A: Manual Python Loop ---
    model_a, opt_a, ctx_a = create_setup()
    
    # A1. Compute GAE
    gae_node = Node(node_id="gae", node_type="PPO_GAE")
    gae_inputs = {
        "batch": batch,
        "next_value": next_value,
        "next_terminated": next_terminated
    }
    gae_out = op_ppo_gae(gae_node, gae_inputs, ctx_a)
    advantages = gae_out["advantages"]
    returns = gae_out["returns"]
    
    # A2. Manual Loop
    train_graph = create_train_graph(config)
    
    # We need to use the SAME indices for both runs to compare results
    # So we force the seed before the loop
    np.random.seed(42)
    
    for epoch in range(config.epochs):
        indices = np.arange(config.rollout_steps)
        np.random.shuffle(indices)
        for start in range(0, config.rollout_steps, config.minibatch_size):
            end = start + config.minibatch_size
            mb_indices = indices[start:end]
            
            minibatch = TransitionBatch(
                obs=batch.obs[mb_indices],
                action=batch.action[mb_indices],
                reward=batch.reward[mb_indices],
                terminated=batch.terminated[mb_indices],
                truncated=batch.truncated[mb_indices],
                next_obs=batch.next_obs[mb_indices],
                done=batch.done[mb_indices],
                log_prob=batch.log_prob[mb_indices],
                value=batch.value[mb_indices],
                advantages=advantages[mb_indices],
                returns=returns[mb_indices]
            )
            
            execute(train_graph, initial_inputs={"traj_in": minibatch}, context=ctx_a)

    params_a = list(model_a.parameters())
    
    # Approach B: Graph Loop
    model_b, opt_b, ctx_b = create_setup()
    update_graph = create_ppo_update_graph(config)
    
    # Mock buffer for the SampleBatch node
    class MockBuffer:
        def get_all(self): return batch
        def __len__(self): return config.rollout_steps
        def clear(self): pass
        def sample_query(self, batch_size, seed=None, **kwargs):
            return batch
        
    buffer_registry = BufferRegistry()
    buffer_registry.register(config.buffer_id, MockBuffer())
    ctx_b.buffer_registry = buffer_registry
    
    # Reset seed to match Approach A's shuffle sequence
    np.random.seed(42)
    
    update_inputs = {
        "next_state": {
            "next_value": next_value,
            "next_terminated": next_terminated
        }
    }
    
    execute(update_graph, initial_inputs=update_inputs, context=ctx_b)
    
    params_b = list(model_b.parameters())
    
    # 3. Compare Results
    for p_a, p_b in zip(params_a, params_b):
        assert torch.allclose(p_a, p_b, atol=1e-6), "Models diverged after update!"
        
    print("Success: Manual loop and Graph loop produced identical results.")

if __name__ == "__main__":
    test_loop_equivalence()
