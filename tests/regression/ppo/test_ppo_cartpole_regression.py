import time
import random
import numpy as np
import torch
import gymnasium as gym
import pytest

from registries import (
    make_ppo_network,
    make_ppo_replay_buffer,
    make_ppo_learner,
)
from actors.action_selectors.selectors import CategoricalSelector
from actors.action_selectors.decorators import PPODecorator
from actors.action_selectors.policy_sources import NetworkPolicySource
from core import PPOEpochIterator

# Module-level marker for regression tests
# Declared just below imports as per README.md
pytestmark = pytest.mark.regression


def evaluate_agent(
    env, agent_network, policy_source, action_selector, device, num_episodes=3
):
    """Evaluate the agent on the environment without exploration."""
    scores = []
    agent_network.eval()
    with torch.inference_mode():
        for _ in range(num_episodes):
            state, info = env.reset()
            episode_score = 0.0
            done = False
            while not done:
                obs_tensor = torch.tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)

                result = policy_source.get_inference(obs=obs_tensor, info=info)
                action, _ = action_selector.select_action(
                    result=result, info=info, exploration=False, )

                state, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated
                episode_score += reward
            scores.append(episode_score)
    agent_network.train()
    return scores


def setup_seeds(seed=42):
    """Setup seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_ppo_cartpole_full_training():
    """
    Heavy full training test for PPO on CartPole-v1.
    Asserts sample efficiency and final performance.
    """
    setup_seeds()

    # --- Hyperparameters ---
    ENV_ID = "CartPole-v1"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    STEPS_PER_EPOCH = 512
    NUM_MINIBATCHES = 4
    TRAIN_POLICY_ITERATIONS = 4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_PARAM = 0.2
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    LEARNING_RATE = 2.5e-4
    TARGET_KL = 0.02
    TOTAL_STEPS = 512000  # Enough to reach high average reliably (450+)

    from components.environment import (
        SimpleEnvObservationComponent,
        SimpleEnvStepComponent,
    )
    from components.actor_logic import (
        NetworkInferenceComponent,
        CategoricalSelectorComponent,
        PPODecoratorComponent,
    )
    from components.memory import BufferStoreComponent
    from core import BlackboardEngine, infinite_ticks

    # --- Setup Environment ---
    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape
    num_actions = env.action_space.n

    # --- Components ---
    agent_network = make_ppo_network(
        obs_dim=obs_dim, num_actions=num_actions, hidden_widths=[64, 64], device=DEVICE, )

    replay_buffer = make_ppo_replay_buffer(
        obs_dim=obs_dim, num_actions=num_actions, steps_per_epoch=STEPS_PER_EPOCH, gamma=GAMMA, gae_lambda=GAE_LAMBDA, )

    optimizer = torch.optim.Adam(agent_network.parameters(), lr=LEARNING_RATE)

    learner = make_ppo_learner(
        agent_network=agent_network, optimizer=optimizer, minibatch_size=STEPS_PER_EPOCH // NUM_MINIBATCHES, num_actions=num_actions, device=DEVICE, clip_param=CLIP_PARAM, entropy_coef=ENTROPY_COEF, value_coef=VALUE_COEF, max_grad_norm=0.5, target_kl=TARGET_KL, )

    # --- Collection Pipeline ---
    obs_comp = SimpleEnvObservationComponent(env)
    
    # Custom field map for PPO buffer keys
    ppo_field_map = {
        "observations": "data.observations",
        "actions": "meta.action",
        "rewards": "data.rewards",
        "dones": "data.dones",
        "values": "meta.action_metadata.value",
        "old_log_probs": "meta.action_metadata.log_prob",
    }
    
    collection_components = [
        obs_comp,
        NetworkInferenceComponent(agent_network, obs_dim),
        CategoricalSelectorComponent(exploration=True),
        PPODecoratorComponent(),
        SimpleEnvStepComponent(env, obs_comp),
        BufferStoreComponent(replay_buffer, field_map=ppo_field_map),
    ]
    collector = BlackboardEngine(collection_components, device=DEVICE)

    # --- Training Loop ---
    steps_collected = 0
    training_scores = []
    state, info = env.reset()

    print("Starting PPO training loop...")
    while steps_collected < TOTAL_STEPS:
        epoch_steps = 0
        trajectory_start_index = replay_buffer.size

        # Collection Phase
        for result in collector.step(infinite_ticks()):
            meta = result["meta"]
            epoch_steps += 1
            steps_collected += 1
            
            if "episode_score" in meta:
                training_scores.append(meta["episode_score"])
                if len(training_scores) % 50 == 0:
                    print(f"Game {len(training_scores)} | Score: {meta['episode_score']} | Avg (L100): {np.mean(training_scores[-100:]):.2f} | Total Steps: {steps_collected}")

            # PPO Trajectory Finishing Logic (at episode end or epoch end)
            if meta["done"] or epoch_steps == STEPS_PER_EPOCH:
                if meta["terminated"]:
                    last_value = 0.0
                else:
                    # Bootstrap value for truncated trajectories
                    with torch.inference_mode():
                        # Try to get final_observation from info (Gym standard for truncated episodes)
                        final_obs = meta.get("info", {}).get("final_observation")
                        if final_obs is not None:
                            obs_t = torch.as_tensor(final_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        else:
                            # Fallback to the current observation in the env_state/obs_comp
                            obs_t = torch.as_tensor(obs_comp.current_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                            
                        out = agent_network.obs_inference(obs_t)
                        last_value = out.value.item()

                trajectory_end_index = replay_buffer.size
                trajectory_slice = slice(trajectory_start_index, trajectory_end_index)

                if trajectory_end_index > trajectory_start_index:
                    res = replay_buffer.input_processor.finish_trajectory(
                        replay_buffer.buffers, trajectory_slice, last_value=last_value, )
                    if res:
                        for k, v in res.items():
                            replay_buffer.buffers[k][trajectory_slice] = v
                
                trajectory_start_index = trajectory_end_index

            if epoch_steps >= STEPS_PER_EPOCH:
                break

        # Learning Phase
        iterator = PPOEpochIterator(
            replay_buffer=replay_buffer, num_epochs=TRAIN_POLICY_ITERATIONS, num_minibatches=NUM_MINIBATCHES, device=DEVICE, )
        for _ in learner.step(iterator):
            pass
        replay_buffer.clear()

        # Early break if solved (475+) to speed up test
        if len(training_scores) >= 100:
            avg_training_score = np.mean(training_scores[-100:])
            if avg_training_score >= 475.0:
                print(f"Solved! Final Avg Training Score (last 100): {avg_training_score:.2f}")
                break

    # --- Assertions ---
    assert len(training_scores) >= 100
    avg_training_score = np.mean(training_scores[-100:])
    assert avg_training_score >= 450.0

    # Evaluation
    policy_source = NetworkPolicySource(agent_network, obs_dim)
    action_selector = CategoricalSelector()
    test_scores = evaluate_agent(
        env, agent_network, policy_source, action_selector, DEVICE, num_episodes=3
    )
    print(f"Evaluation scores: {test_scores}")
    for i, score in enumerate(test_scores):
        assert score == 500.0

    env.close()


if __name__ == "__main__":
    test_ppo_cartpole_full_training()
    print("Test passed!")