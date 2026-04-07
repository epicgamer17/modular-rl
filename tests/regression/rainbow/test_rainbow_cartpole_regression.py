import random
import numpy as np
import torch
import gymnasium as gym
import pytest

pytestmark = pytest.mark.regression

import time

from modules.agent_nets.modular import ModularAgentNetwork
from modules.backbones.mlp import MLPBackbone
from modules.heads.q import QHead
from learner.losses.representations import C51Representation
from actors.action_selectors.selectors import ArgmaxSelector

from data.storage.circular import BufferConfig, ModularReplayBuffer
from data.concurrency import LocalBackend
from data.processors import (
    StackedInputProcessor,
    NStepInputProcessor,
    TerminationFlagsInputProcessor,
    LegalMovesMaskProcessor,
    FilterKeysInputProcessor,
    StandardOutputProcessor,
)
from data.writers import CircularWriter
from data.samplers.prioritized import PrioritizedSampler

from learner.pipeline.base import UniversalLearner
from learner.losses.loss_pipeline import LossPipeline
from learner.losses.q import QBootstrappingLoss
from learner.losses.priorities import MaxLossPriorityComputer
from learner.pipeline.target_builders import (
    DistributionalTargetBuilder,
    TargetBuilderPipeline,
    SingleStepFormatter,
)
from learner.pipeline.batch_iterators import RepeatSampleIterator


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

    # --- Setup Environment ---
    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape
    num_actions = env.action_space.n

    # --- Components ---
    # 1. Network
    representation = C51Representation(vmin=V_MIN, vmax=V_MAX, bins=ATOM_SIZE)

    def make_network():
        backbone = MLPBackbone(input_shape=obs_dim, widths=[128])
        head = QHead(
            input_shape=backbone.output_shape,
            num_actions=num_actions,
            representation=representation,
            hidden_widths=[],  # Minimal head as per MLP widths [128]
            noisy_sigma=NOISY_SIGMA,
        )
        return ModularAgentNetwork(
            components={
                "feature_block": backbone,
                "q_head": head,
            },
            atom_size=ATOM_SIZE,
        ).to(DEVICE)

    agent_network = make_network()
    target_network = make_network()
    target_network.load_state_dict(agent_network.state_dict())
    target_network.eval()

    # 2. Replay Buffer
    buffer_configs = [
        BufferConfig("observations", shape=obs_dim, dtype=torch.float32),
        BufferConfig("actions", shape=(), dtype=torch.int64),
        BufferConfig("rewards", shape=(), dtype=torch.float32),
        BufferConfig("next_observations", shape=obs_dim, dtype=torch.float32),
        BufferConfig("terminated", shape=(), dtype=torch.bool),
        BufferConfig("truncated", shape=(), dtype=torch.bool),
        BufferConfig("dones", shape=(), dtype=torch.bool),
        BufferConfig("next_legal_moves_masks", shape=(num_actions,), dtype=torch.bool),
    ]

    input_processor = StackedInputProcessor(
        [
            TerminationFlagsInputProcessor(),
            NStepInputProcessor(n_step=N_STEP, gamma=GAMMA),
            LegalMovesMaskProcessor(
                num_actions,
                input_key="next_legal_moves",
                output_key="next_legal_moves_masks",
            ),
            FilterKeysInputProcessor(
                [
                    "observations",
                    "actions",
                    "rewards",
                    "next_observations",
                    "terminated",
                    "truncated",
                    "dones",
                    "next_legal_moves_masks",
                ]
            ),
        ]
    )

    sampler = PrioritizedSampler(
        max_size=REPLAY_BUFFER_SIZE,
        alpha=PER_ALPHA,
        beta=PER_BETA,
        epsilon=PER_EPSILON,
    )

    replay_buffer = ModularReplayBuffer(
        max_size=REPLAY_BUFFER_SIZE,
        batch_size=MINIBATCH_SIZE,
        buffer_configs=buffer_configs,
        input_processor=input_processor,
        output_processor=StandardOutputProcessor(),
        writer=CircularWriter(REPLAY_BUFFER_SIZE),
        sampler=sampler,
        backend=LocalBackend(),
    )

    # 3. Learner
    optimizer = {
        "default": torch.optim.Adam(
            agent_network.parameters(),
            lr=LEARNING_RATE,
            eps=ADAM_EPSILON,
            weight_decay=WEIGHT_DECAY,
        )
    }

    loss_pipeline = LossPipeline(
        modules=[QBootstrappingLoss(device=DEVICE, is_categorical=True)],
        priority_computer=MaxLossPriorityComputer(loss_key="QBootstrappingLoss"),
        minibatch_size=MINIBATCH_SIZE,
        atom_size=ATOM_SIZE,
        representations={"q_logits": representation},
    )

    target_builder = TargetBuilderPipeline(
        [
            DistributionalTargetBuilder(
                target_network=target_network,
                gamma=GAMMA,
                n_step=N_STEP,
            ),
            SingleStepFormatter(),
        ]
    )

    from learner.pipeline.callbacks import PriorityUpdaterCallback, ResetNoiseCallback
    from utils.schedule import ConstantSchedule

    per_beta_schedule = ConstantSchedule(PER_BETA)

    callbacks = [
        PriorityUpdaterCallback(
            priority_update_fn=replay_buffer.update_priorities,
            set_beta_fn=replay_buffer.set_beta,
            per_beta_schedule=per_beta_schedule,
        ),
        ResetNoiseCallback(),
    ]

    learner = UniversalLearner(
        agent_network=agent_network,
        device=DEVICE,
        num_actions=num_actions,
        observation_dimensions=obs_dim,
        observation_dtype=torch.float32,
        loss_pipeline=loss_pipeline,
        optimizer=optimizer,
        target_builder=target_builder,
        clipnorm=CLIP_NORM,
        callbacks=callbacks,
    )

    # --- Training Loop ---
    training_scores = []
    global_step = 0
    state, info = env.reset()
    current_episode_score = 0.0

    print(
        f"Starting Rainbow CartPole regression training for {TRAINING_STEPS} learning steps..."
    )

    # 1. Warm-up Phase: Fill buffer before training starts
    while global_step < MIN_REPLAY_SIZE:
        with torch.no_grad():
            obs_tensor = torch.tensor(
                state, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)
            result = agent_network.obs_inference(obs_tensor)
            q_values = result.q_values
            action_val = q_values.argmax(dim=-1).item()

        next_state, reward, terminated, truncated, next_info = env.step(action_val)
        done = terminated or truncated
        current_episode_score += reward

        replay_buffer.store(
            observations=state,
            actions=action_val,
            rewards=reward,
            next_observations=next_state,
            dones=done,
            next_legal_moves=next_info.get("legal_moves"),
        )

        if done:
            training_scores.append(current_episode_score)
            state, info = env.reset()
            current_episode_score = 0.0
        else:
            state, info = next_state, next_info
        global_step += 1

    # 2. Main Training Loop: Replay Interval steps -> 1 Learning step
    for learning_step in range(1, TRAINING_STEPS + 1):
        # Step environment REPLAY_INTERVAL times
        for _ in range(REPLAY_INTERVAL):
            with torch.no_grad():
                obs_tensor = torch.tensor(
                    state, dtype=torch.float32, device=DEVICE
                ).unsqueeze(0)
                result = agent_network.obs_inference(obs_tensor)
                q_values = result.q_values
                action_val = q_values.argmax(dim=-1).item()

            next_state, reward, terminated, truncated, next_info = env.step(action_val)
            done = terminated or truncated
            current_episode_score += reward

            replay_buffer.store(
                observations=state,
                actions=action_val,
                rewards=reward,
                next_observations=next_state,
                dones=done,
                next_legal_moves=next_info.get("legal_moves"),
            )

            state, info = next_state, next_info
            global_step += 1

            if done:
                training_scores.append(current_episode_score)
                state, info = env.reset()
                current_episode_score = 0.0

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
            print(
                f"Learning Step {learning_step} | Total Steps {global_step} | Avg Score: {avg_score:.2f}"
            )

            # Early stop if we reach the goal consistently (480.0 is near-perfect for CartPole)
            if len(training_scores) >= 10 and np.mean(training_scores[-10:]) >= 490.0:
                print(f"Goal reached at learning step {learning_step}!")
                break

    # --- Final Evaluation ---
    test_scores = evaluate_agent(env, agent_network, DEVICE, num_episodes=100)
    avg_test_score = np.mean(test_scores)
    print(f"Final Test Scores: {test_scores} | Avg: {avg_test_score:.2f}")

    # --- Assertions ---
    assert len(training_scores) > 0, "No episodes completed during training"
    final_avg = np.mean(training_scores[-100:])
    assert final_avg >= 400.0, f"Average training score {final_avg:.2f} is below 400.0"
    assert (
        avg_test_score >= 450.0
    ), f"Average test score {avg_test_score:.2f} is below 450.0"

    env.close()


if __name__ == "__main__":
    test_rainbow_cartpole_full_training()
