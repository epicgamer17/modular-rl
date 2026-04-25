import pytest
import torch
from unittest.mock import MagicMock, patch
from agents.ppo.learner import OnPolicyLearner
from agents.ppo.config import PPOConfig
from runtime.context import ExecutionContext
from runtime.state import ModelRegistry, OptimizerRegistry, BufferRegistry
from agents.ppo.buffer import RolloutBuffer

pytestmark = pytest.mark.unit

def test_kl_no_stop_when_small():
    """
    Small KL runs all epochs.
    """
    # 1. Setup Config
    epochs = 5
    target_kl = 0.01
    config = PPOConfig(
        obs_dim=4,
        act_dim=2,
        epochs=epochs,
        target_kl=target_kl,
        rollout_steps=10,
        minibatch_size=5, # 2 minibatches per epoch
        num_envs=1
    )
    
    # 2. Setup Components
    from agents.ppo.model import ActorCritic
    model = ActorCritic(obs_dim=4, act_dim=2)
    
    model_registry = ModelRegistry()
    model_registry.register(config.model_handle, model)

    optimizer_registry = OptimizerRegistry()
    from torch.optim import Adam
    from runtime.state import OptimizerState
    opt_state = OptimizerState(Adam(model.parameters(), lr=config.learning_rate))
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
    
    # Fill buffer enough to trigger update
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
    
    # 4. Mock super().update_step to return small KL
    # Each call returns approx_kl = 0.005 ( < target_kl 0.01)
    with patch("runtime.engine.LearnerRuntime.update_step") as mock_update:
        mock_update.return_value = {"opt": {"approx_kl": 0.005}}
        
        learner.update_step(context=ctx)
        
        # It should run all 5 epochs. 2 minibatches * 5 epochs = 10 calls.
        assert mock_update.call_count == 10, f"Expected 10 calls (5 epochs), got {mock_update.call_count}"

if __name__ == "__main__":
    test_kl_no_stop_when_small()
