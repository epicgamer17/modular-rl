import pytest
import gymnasium as gym
from agents.ppo.config import PPOConfig
from agents.ppo.model import ActorCritic
from agents.ppo.operators import register_ppo_operators
from agents.ppo.graphs import create_interact_graph, create_train_graph
from agents.ppo.rollout import create_ppo_recording_fn
from agents.ppo.learner import OnPolicyLearner
from agents.ppo.metrics import PPOMetrics
from agents.ppo.agent import PPOAgent

pytestmark = pytest.mark.unit

def test_ppo_modules_importable():
    """Verify all PPO modules are importable without circular dependencies."""
    # This test passes if the imports above succeed
    assert True

def test_ppo_agent_initialization():
    """Verify PPOAgent can be initialized."""
    config = PPOConfig(
        obs_dim=4,
        act_dim=2,
        hidden_dim=16,
        rollout_steps=128,
        num_envs=1,
        minibatch_size=64
    )
    env = gym.make("CartPole-v1")
    agent = PPOAgent(config, env)
    
    assert agent.ac_net is not None
    assert agent.interact_graph is not None
    assert agent.train_graph is not None
    assert agent.executor is not None

def test_ppo_operator_registration():
    """Verify PPO operators can be registered."""
    from runtime.operator_registry import OPERATOR_REGISTRY
    register_ppo_operators()
    
    assert "PPO_PolicyActor" in OPERATOR_REGISTRY
    assert "PPO_GAE" in OPERATOR_REGISTRY
    assert "PPO_Objective" in OPERATOR_REGISTRY
    assert "PPO_Optimizer" in OPERATOR_REGISTRY
