"""
PPO Implementation using the modular PPO agent.
Demonstrates on-policy scheduling, explicit ports, and model registries.
"""

import gymnasium as gym
from agents.ppo import PPOAgent, PPOConfig
from envs.wrappers import NormalizeObservation


def run_ppo_demo(total_steps=500_000):
    # PPO is most efficient with vectorized environments
    # TODO: Make PPO work with Multiple Envs
    num_envs = 1  # not sure if i love or not

    # Create vectorized gym env
    def make_env():
        return gym.make("CartPole-v1")

    raw_env = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
    # Add normalization wrapper
    raw_env = NormalizeObservation(raw_env)

    from runtime.environment import wrap_env

    env = wrap_env(raw_env)

    obs_dim = env.obs_spec.shape[0]
    act_dim = 2  # CartPole-v1 has 2 actions

    # 1. Define Config
    config = PPOConfig(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=64,
        rollout_steps=512,  # 128 * 4 Envs
        num_envs=num_envs,
        minibatch_size=128,
        epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        learning_rate=2.5e-4,
        adam_epsilon=1e-5,
        target_kl=0.01,  # None
        max_grad_norm=0.5,
        anneal_lr=False,  # True
        normalize_advantages=True,
        total_steps=total_steps,
    )

    # 2. Initialize Modular Agent
    agent = PPOAgent(config, env)

    # 3. Add Logging to Recording Function
    base_record = agent.actor_runtime.recording_fn

    def logging_record(single_step):
        # Ensure the actual PPO recording (to buffer) still happens
        if base_record:
            base_record(single_step)

        # Check if the environment finished
        if single_step.done:
            # ActorRuntime maintains the return of the most recently finished episode
            # metadata is a dict inside TransitionBatch
            step_idx = single_step.metadata["step_index"]
            print(
                f"Step {step_idx} | Episode Return: {agent.actor_runtime.last_episode_return:.2f}"
            )

    agent.actor_runtime.recording_fn = logging_record

    # 4. Train
    agent.train(total_steps=total_steps)
    print("PPO Modern Demo Finished.")


if __name__ == "__main__":
    run_ppo_demo()
