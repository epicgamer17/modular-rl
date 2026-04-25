import pytest
import torch
import numpy as np
from runtime.bootstrap import bootstrap_runtime
from runtime.registry import clear_registry

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def setup():
    clear_registry()
    bootstrap_runtime()


def test_dqn_deterministic_smoke():
    """
    Verifies that running DQN with the same seed produces the same results.
    We capture the weights after a few steps and compare them.
    """
    # Use a small number of steps for the smoke test
    num_steps = 100
    seed = 42
    
    # Run first time
    torch.manual_seed(seed)
    np.random.seed(seed)
    # We need a way to capture results without printing everything
    # For now, we'll just run it and check if it finishes without error,
    # but to really check determinism we should compare weights.
    
    from agents.dqn.config import DQNConfig
    from agents.dqn.agent import DQNAgent
    from runtime.engine import ActorRuntime, LearnerRuntime
    from runtime.runner import ScheduleRunner
    from compiler.planner import compile_schedule
    import gymnasium as gym

    def run_minimal_dqn(steps, s):
        torch.manual_seed(s)
        np.random.seed(s)
        env = gym.make("CartPole-v1")
        env.reset(seed=s)
        
        config = DQNConfig(
            obs_dim=4, act_dim=2, hidden_dim=8, 
            min_replay_size=10, batch_size=4,
            epsilon_decay_steps=50
        )
        agent = DQNAgent(config)
        agent.compile(strict=True)
        
        ctx = agent.get_execution_context(seed=s)
        actor_rt = ActorRuntime(agent.actor_graph, env, replay_buffer=agent.rb)
        learner_rt = LearnerRuntime(agent.learner_graph, replay_buffer=agent.rb)
        
        plan = compile_schedule(agent.learner_graph, user_hints={"actor_frequency": 1, "learner_frequency": 1})
        executor = ScheduleRunner(plan, actor_rt, learner_rt)
        executor.run(total_actor_steps=steps, context=ctx)
        
        # Return weights for comparison
        return [p.detach().clone() for p in agent.q_net.parameters()]

    # Run twice
    weights1 = run_minimal_dqn(num_steps, seed)
    weights2 = run_minimal_dqn(num_steps, seed)
    
    # Compare weights
    for w1, w2 in zip(weights1, weights2):
        assert torch.allclose(w1, w2), "Weights diverged between two runs with same seed!"
    
    # Run with different seed
    weights3 = run_minimal_dqn(num_steps, seed + 1)
    
    # Compare weights (should be different)
    all_match = True
    for w1, w3 in zip(weights1, weights3):
        if not torch.allclose(w1, w3):
            all_match = False
            break
    assert not all_match, "Weights were identical even with different seeds!"

    print("\n[Success] DQN is deterministic across identical seeds and stochastic across different seeds.")
