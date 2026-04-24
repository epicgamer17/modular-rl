"""
PPO Implementation using the modular PPO agent.
Demonstrates on-policy scheduling, explicit ports, and model registries.
"""

import gymnasium as gym
from agents.ppo import PPOAgent, PPOConfig
from envs.wrappers import NormalizeObservation


def run_ppo_demo(total_steps=256_000):
    env = gym.make("CartPole-v1")
    env = NormalizeObservation(env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 1. Define Config
    config = PPOConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=64,
        rollout_steps=512,
        num_envs=1,
        minibatch_size=128,
        epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        learning_rate=3e-4,
        target_kl=0.03,
        max_grad_norm=0.5,
        anneal_lr=True,
        normalize_advantages=True,
    )

    # 2. Initialize Modular Agent
    agent = PPOAgent(config, env)

    # 3. Train
    agent.train(total_steps=total_steps)
    print("PPO Modern Demo Finished.")


if __name__ == "__main__":
    run_ppo_demo()
