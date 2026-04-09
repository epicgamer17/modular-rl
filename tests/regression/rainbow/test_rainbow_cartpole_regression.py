import random
import numpy as np
import torch
import gymnasium as gym
import pytest

pytestmark = pytest.mark.regression

import time

from registries import (
    make_rainbow_network,
    make_rainbow_replay_buffer,
    make_rainbow_learner,
)
from actors.action_selectors.selectors import ArgmaxSelector
from core import RepeatSampleIterator
from utils.schedule import ConstantSchedule

# Module-level marker for regression tests
pytestmark = [pytest.mark.regression, pytest.mark.slow]


def setup_seeds(seed=42):
    """Setup seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_agent(env, agent_network, device, num_episodes=3):
    """Evaluate the agent on the environment using greedy actions."""
    scores = []
    agent_network.eval()
    action_selector = ArgmaxSelector()
    with torch.inference_mode():
        for _ in range(num_episodes):
            state, info = env.reset()
            episode_score = 0.0
            done = False
            while not done:
                obs_tensor = torch.tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)
                # Rainbow inference via obs_inference returns InferenceOutput with q_values
                result = agent_network.obs_inference(obs_tensor)
                action, _ = action_selector.select_action(result=result, info=info)
                state, reward, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated
                episode_score += reward
            scores.append(episode_score)
    agent_network.train()
    return scores


@pytest.mark.xfail(reason="No hyperparams")
def test_rainbow_cartpole_full_training():
    """
    Standalone regression test for Rainbow DQN on CartPole-v1.
    Matches the provided hyperparameters and logic.
    """
    setup_seeds()

    # --- Hyperparameters ---
    ENV_ID = "CartPole-v1"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: CURT PARK PARITY
    LEARNING_RATE = 0.001
    ADAM_EPSILON = 1e-8
    TRAINING_STEPS = 10000
    MINIBATCH_SIZE = 128
    TRANSFER_INTERVAL = 100
    REPLAY_INTERVAL = 4
    N_STEP = 3
    GAMMA = 0.99
    CLIP_NORM = 10.0
    NOISY_SIGMA = 0.5
    ATOM_SIZE = 51
    V_MIN = 0.0
    V_MAX = 500.0
    WEIGHT_DECAY = 0.0

    PER_ALPHA = 0.2
    PER_BETA = 0.6
    PER_EPSILON = 1e-6
    REPLAY_BUFFER_SIZE = 100000
    MIN_REPLAY_SIZE = MINIBATCH_SIZE + N_STEP

    from components.environment import (
        SimpleEnvObservationComponent,
        SimpleEnvStepComponent,
    )
    from components.actor_logic import (
        NetworkInferenceComponent,
        ArgmaxSelectorComponent,
    )
    from components.memory import BufferStoreComponent
    from core import BlackboardEngine, infinite_ticks
    from actors.action_selectors.policy_sources import NetworkPolicySource

    # --- Setup Environment ---
    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape
    num_actions = env.action_space.n

    # --- Components ---
    # 1. Network
    agent_network = make_rainbow_network(
        obs_dim=obs_dim, num_actions=num_actions, hidden_widths=[128], noisy_sigma=NOISY_SIGMA, atom_size=ATOM_SIZE, v_min=V_MIN, v_max=V_MAX, device=DEVICE, )
    target_network = make_rainbow_network(
        obs_dim=obs_dim, num_actions=num_actions, hidden_widths=[128], noisy_sigma=NOISY_SIGMA, atom_size=ATOM_SIZE, v_min=V_MIN, v_max=V_MAX, device=DEVICE, )
    target_network.load_state_dict(agent_network.state_dict())
    target_network.eval()

    # 2. Replay Buffer
    replay_buffer = make_rainbow_replay_buffer(
        obs_dim=obs_dim, num_actions=num_actions, max_size=REPLAY_BUFFER_SIZE, batch_size=MINIBATCH_SIZE, n_step=N_STEP, gamma=GAMMA, per_alpha=PER_ALPHA, per_beta=PER_BETA, per_epsilon=PER_EPSILON, )

    # 3. Learner
    optimizer = torch.optim.Adam(
        agent_network.parameters(), lr=LEARNING_RATE, eps=ADAM_EPSILON, weight_decay=WEIGHT_DECAY, )

    per_beta_schedule = ConstantSchedule(PER_BETA)

    learner = make_rainbow_learner(
        agent_network=agent_network, target_network=target_network, optimizer=optimizer, replay_buffer=replay_buffer, gamma=GAMMA, n_step=N_STEP, clip_norm=CLIP_NORM, per_beta_schedule=per_beta_schedule, device=DEVICE, )

    # --- Pipelines ---
    obs_comp = SimpleEnvObservationComponent(env)
    
    # Collection Pipeline
    collection_components = [
        obs_comp,
        NetworkInferenceComponent(agent_network, obs_dim),
        ArgmaxSelectorComponent(),
        SimpleEnvStepComponent(env, obs_comp),
        BufferStoreComponent(replay_buffer),
    ]
    collector = BlackboardEngine(collection_components, device=DEVICE)

    # Evaluation Pipeline
    eval_obs_comp = SimpleEnvObservationComponent(env)
    eval_components = [
        eval_obs_comp,
        NetworkInferenceComponent(agent_network, obs_dim),
        ArgmaxSelectorComponent(),
        SimpleEnvStepComponent(env, eval_obs_comp, stop_on_done=True),
    ]
    evaluator = BlackboardEngine(eval_components, device=DEVICE)

    # --- Training Loop ---
    training_scores = []
    global_step = 0

    print(f"Starting Rainbow CartPole regression training for {TRAINING_STEPS} learning steps...")

    # 1. Warm-up Phase: Fill buffer before training starts
    print("Warm-up phase...")
    for _ in collector.step(infinite_ticks()):
        global_step += 1
        if global_step >= MIN_REPLAY_SIZE:
            break

    # 2. Main Training Loop: Replay Interval steps -> 1 Learning step
    for learning_step in range(1, TRAINING_STEPS + 1):
        # Step environment REPLAY_INTERVAL times
        for _ in range(REPLAY_INTERVAL):
            # Run one step of collection
            result = next(collector.step(infinite_ticks()))
            global_step += 1
            
            if "episode_score" in result["meta"]:
                training_scores.append(result["meta"]["episode_score"])

            # Target Network Sync (based on global_step)
            if global_step % TRANSFER_INTERVAL == 0:
                target_network.load_state_dict(agent_network.state_dict())

        # Perform 1 learning step
        iterator = RepeatSampleIterator(replay_buffer, num_iterations=1, device=DEVICE)
        for _ in learner.step(iterator):
            pass

        # Logging every 100 learning steps
        if learning_step % 100 == 0:
            avg_score = np.mean(training_scores[-100:]) if training_scores else 0.0
            print(f"Learning Step {learning_step} | Total Steps {global_step} | Avg Score: {avg_score:.2f}")

            # Early stop if we reach the goal consistently (480.0 is near-perfect for CartPole)
            if len(training_scores) >= 10 and np.mean(training_scores[-10:]) >= 490.0:
                print(f"Goal reached at learning step {learning_step}!")
                break

    # --- Final Evaluation ---
    print("Final Evaluation...")
    agent_network.eval()
    test_scores = []
    for _ in range(100):
        # Run until stop_on_done
        for result in evaluator.step(infinite_ticks()):
            if "episode_score" in result["meta"]:
                test_scores.append(result["meta"]["episode_score"])
                break
    
    avg_test_score = np.mean(test_scores)
    print(f"Final Test Scores: Average {avg_test_score:.2f}")
    agent_network.train()

    # --- Assertions ---
    assert len(training_scores) > 0, "No episodes completed during training"
    final_avg = np.mean(training_scores[-100:])
    assert final_avg >= 400.0, f"Average training score {final_avg:.2f} is below 400.0"
    assert avg_test_score >= 450.0, f"Average test score {avg_test_score:.2f} is below 450.0"

    env.close()


if __name__ == "__main__":
    test_rainbow_cartpole_full_training()