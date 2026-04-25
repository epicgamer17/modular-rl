"""
DQN Implementation using the modular agents system.
Demonstrates off-policy learning using DQNAgent from agents/dqn/.
"""

import torch
import numpy as np
import gymnasium as gym
from agents.dqn.config import DQNConfig
from agents.dqn.agent import DQNAgent
from runtime.engine import ActorRuntime, LearnerRuntime
from runtime.runner import ScheduleRunner
from compiler.planner import compile_schedule


# TODO: make total steps in learner steps and not in env steps (or able to choose which)
def train_dqn(total_steps: int = 120_000, seed: int = 0):
    # 0. Global Seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True  # For maximum reproducibility

    # 1. Environment Setup
    from runtime.io.environment import wrap_env

    raw_env = gym.make("CartPole-v1")
    env = wrap_env(raw_env)
    env.reset(seed=seed)

    obs_dim = env.obs_spec.shape[0]
    act_dim = 2  # CartPole-v1

    # 2. Configuration
    config = DQNConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=512,
        lr=1e-3,
        gamma=0.99,
        buffer_capacity=50000,
        batch_size=128,
        min_replay_size=100,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=1000,
        target_sync_frequency=100,
    )

    # 3. Agent Initialization
    # This automatically handles model creation, buffer setup, and graph building
    agent = DQNAgent(config)

    # 4. Compilation
    # Validates the IR graphs and prepares them for execution
    agent.compile(strict=True)

    # 5. Runtime Setup
    # Create runtimes using the agent's graphs and state
    actor_runtime = ActorRuntime(agent.actor_graph, env, replay_buffer=agent.rb)

    learner_runtime = LearnerRuntime(agent.learner_graph, replay_buffer=agent.rb)

    # 6. Scheduling
    # Compile a schedule that interleaves actor and learner steps
    plan = compile_schedule(
        agent.learner_graph,  # Schedule is derived from the learner's needs
        user_hints={
            "actor_frequency": 4,
            "learner_frequency": 1,
        },
    )

    # 7. Execution
    from observability.dispatcher import setup_default_observability
    setup_default_observability()

    ctx = agent.get_execution_context(seed=seed)

    runner = ScheduleRunner(plan, actor_runtime, learner_runtime)


    print(f"Starting DQN with Modular Agent and Compiled Schedule: {plan.to_dict()}")
    try:
        runner.run(total_actor_steps=total_steps, context=ctx)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Generating plots...")
    finally:
        print("DQN Modular Demo Finished.")
        
        # 8. Plot Results
        from observability.plotting.rl_plots import plot_metric
        plot_metric("episode_return", title="DQN: Episodic Return", save_path="dqn_return.png")
        plot_metric("loss", title="DQN: Bellman Loss", save_path="dqn_loss.png")
        plot_metric("sps", title="DQN: Training Throughput (SPS)", save_path="dqn_sps.png")
        print("Plots saved to dqn_return.png, dqn_loss.png, and dqn_sps.png")



if __name__ == "__main__":
    train_dqn()
