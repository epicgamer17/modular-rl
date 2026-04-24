import pytest
import torch
import torch.optim as optim
from agents.ppo.learner import OnPolicyLearner
from agents.ppo.config import PPOConfig
from runtime.context import ExecutionContext
from runtime.state import ModelRegistry, OptimizerRegistry, OptimizerState, BufferRegistry
from agents.ppo.buffer import RolloutBuffer

pytestmark = pytest.mark.unit

def test_lr_anneals():
    """
    Test that the learning rate anneals linearly based on actor steps.
    """
    # 1. Setup Config
    total_steps = 1000
    initial_lr = 1e-3
    config = PPOConfig(
        obs_dim=4,
        act_dim=2,
        total_steps=total_steps,
        learning_rate=initial_lr,
        anneal_lr=True,
        rollout_steps=10,
        minibatch_size=10,
        num_envs=1
    )
    
    # 2. Setup Components
    from agents.ppo.model import ActorCritic
    model = ActorCritic(obs_dim=4, act_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    opt_state = OptimizerState(optimizer)
    
    model_registry = ModelRegistry()
    model_registry.register(config.model_handle, model)
    
    optimizer_registry = OptimizerRegistry()
    optimizer_registry.register(config.optimizer_handle, opt_state)
    
    buffer = RolloutBuffer(rollout_steps=10, num_envs=1, obs_dim=4)
    buffer_registry = BufferRegistry()
    buffer_registry.register(config.buffer_id, buffer)
    
    # Mock Actor Runtime to provide last_obs/last_done
    class MockActorRuntime:
        def __init__(self):
            self.last_obs = torch.zeros(4)
            self.last_done = torch.zeros(1)
            self.last_terminated = torch.zeros(1)
            self.last_episode_return = 0.0
            self.last_episode_length = 0
            
    actor_runtime = MockActorRuntime()
    
    # 3. Setup Learner
    # We need a dummy train_graph
    from core.graph import Graph
    train_graph = Graph()
    
    learner = OnPolicyLearner(
        train_graph=train_graph,
        config=config,
        ac_net=model,
        actor_runtime=actor_runtime,
        buffer_id=config.buffer_id
    )
    
    ctx = ExecutionContext(
        model_registry=model_registry,
        optimizer_registry=optimizer_registry,
        buffer_registry=buffer_registry
    )
    
    # 4. Fill buffer enough to trigger update
    def fill_buffer():
        for _ in range(10):
            buffer.add(
                torch.zeros(1, 4), # obs
                torch.zeros(1),    # action
                torch.zeros(1),    # reward
                torch.zeros(1),    # terminated
                torch.zeros(1),    # truncated
                torch.zeros(1),    # value
                torch.zeros(1),    # log_prob
                0                  # policy_version
            )
    
    fill_buffer()
    
    # 5. Check LR at different steps
    # At step 0
    ctx.actor_step = 0
    learner.update_step(context=ctx)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr)
    
    # At step 500 (halfway)
    ctx.actor_step = 500
    # Refill buffer as it was cleared
    fill_buffer()
    learner.update_step(context=ctx)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr * 0.5)
    
    # At step 1000 (end)
    ctx.actor_step = 1000
    # Refill buffer
    fill_buffer()
    learner.update_step(context=ctx)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
