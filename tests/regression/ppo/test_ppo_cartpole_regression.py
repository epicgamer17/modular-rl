import time
import random
import numpy as np
import torch
import gymnasium as gym
import pytest

from modules.agent_nets.modular import ModularAgentNetwork
from modules.backbones.mlp import MLPBackbone
from modules.heads.policy import PolicyHead
from modules.heads.value import ValueHead
from learner.losses.representations import (
    ClassificationRepresentation,
    ScalarRepresentation,
)
from actors.action_selectors.selectors import CategoricalSelector
from actors.action_selectors.decorators import PPODecorator
from actors.action_selectors.policy_sources import NetworkPolicySource

from data.storage.circular import BufferConfig, ModularReplayBuffer
from data.concurrency import LocalBackend
from data.processors import (
    GAEProcessor,
    LegalMovesMaskProcessor,
    AdvantageNormalizer,
    StackedInputProcessor,
)
from data.writers import PPOWriter
from data.samplers.prioritized import WholeBufferSampler

from learner.pipeline.batch_iterators import PPOEpochIterator
from learner.base import UniversalLearner
from learner.losses.loss_pipeline import LossPipeline
from learner.losses.policy import ClippedSurrogateLoss
from learner.losses.value import ClippedValueLoss
from learner.pipeline.callbacks import MetricEarlyStopCallback
from learner.pipeline.targets import (
    PassThroughTargetBuilder,
    TargetBuilderPipeline,
    SingleStepFormatter,
    TargetFormatter,
)
from learner.losses.shape_validator import ShapeValidator

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
                    result=result,
                    info=info,
                    exploration=False,
                )

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

    # --- Setup Environment ---
    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape
    num_actions = env.action_space.n

    # --- Components ---
    agent_network = ModularAgentNetwork(
        components={
            "policy_head": PolicyHead(
                input_shape=obs_dim,
                representation=ClassificationRepresentation(num_classes=num_actions),
                neck=MLPBackbone(input_shape=obs_dim, widths=[64, 64]),
            ),
            "value_head": ValueHead(
                input_shape=obs_dim,
                representation=ScalarRepresentation(),
                neck=MLPBackbone(input_shape=obs_dim, widths=[64, 64]),
            ),
        },
    ).to(DEVICE)

    action_selector = CategoricalSelector(exploration=True)
    action_selector = PPODecorator(inner_selector=action_selector)
    policy_source = NetworkPolicySource(agent_network, input_shape=obs_dim)

    configs = [
        BufferConfig("observations", shape=obs_dim, dtype=torch.float32),
        BufferConfig("actions", shape=(), dtype=torch.int64),
        BufferConfig("rewards", shape=(), dtype=torch.float32),
        BufferConfig("values", shape=(), dtype=torch.float32),
        BufferConfig("old_log_probs", shape=(), dtype=torch.float32),
        BufferConfig("legal_moves_masks", shape=(num_actions,), dtype=torch.bool),
        BufferConfig("advantages", shape=(), dtype=torch.float32),
        BufferConfig("returns", shape=(), dtype=torch.float32),
    ]

    input_stack = StackedInputProcessor(
        [
            GAEProcessor(GAMMA, GAE_LAMBDA),
            LegalMovesMaskProcessor(
                num_actions, input_key="legal_moves", output_key="legal_moves_masks"
            ),
        ]
    )

    replay_buffer = ModularReplayBuffer(
        max_size=STEPS_PER_EPOCH,
        batch_size=STEPS_PER_EPOCH,
        buffer_configs=configs,
        input_processor=input_stack,
        output_processor=AdvantageNormalizer(),
        writer=PPOWriter(STEPS_PER_EPOCH),
        sampler=WholeBufferSampler(),
        backend=LocalBackend(),
    )

    optimizer = {
        "default": torch.optim.Adam(agent_network.parameters(), lr=LEARNING_RATE)
    }

    pol_rep = agent_network.components["policy_head"].representation
    val_rep = agent_network.components["value_head"].representation

    minibatch_size = STEPS_PER_EPOCH // NUM_MINIBATCHES
    shape_validator = ShapeValidator(
        minibatch_size=minibatch_size,
        num_actions=num_actions,
        unroll_steps=0,
    )

    loss_pipeline = LossPipeline(
        modules=[
            ClippedSurrogateLoss(
                device=DEVICE,
                clip_param=CLIP_PARAM,
                entropy_coefficient=ENTROPY_COEF,
                optimizer_name="default",
            ),
            ClippedValueLoss(
                device=DEVICE,
                clip_param=CLIP_PARAM,
                target_key="returns",
                optimizer_name="default",
                loss_factor=VALUE_COEF,
            ),
        ],
        minibatch_size=minibatch_size,
        num_actions=num_actions,
        unroll_steps=0,
        representations={"policies": pol_rep, "values": val_rep},
        shape_validator=shape_validator,
    )

    target_builder = TargetBuilderPipeline(
        [
            PassThroughTargetBuilder(
                ["values", "returns", "actions", "old_log_probs", "advantages"]
            ),
            TargetFormatter({"values": val_rep, "returns": val_rep}),
            SingleStepFormatter(),
        ]
    )

    learner = UniversalLearner(
        agent_network=agent_network,
        device=DEVICE,
        num_actions=num_actions,
        observation_dimensions=obs_dim,
        observation_dtype=torch.float32,
        loss_pipeline=loss_pipeline,
        optimizer=optimizer,
        lr_scheduler={},
        target_builder=target_builder,
        callbacks=[MetricEarlyStopCallback(threshold=TARGET_KL)],
        clipnorm=0.5,
        shape_validator=shape_validator,
    )

    # --- Training Loop ---
    steps_collected = 0
    training_scores = []
    current_episode_score = 0.0
    state, info = env.reset()

    print("Starting PPO training loop...")
    while steps_collected < TOTAL_STEPS:
        epoch_steps = 0
        trajectory_start_index = replay_buffer.size

        while epoch_steps < STEPS_PER_EPOCH:
            with torch.no_grad():
                obs_tensor = torch.tensor(
                    state, dtype=torch.float32, device=DEVICE
                ).unsqueeze(0)

                result = policy_source.get_inference(obs=obs_tensor, info=info)
                action, metadata = action_selector.select_action(
                    result=result,
                    info=info,
                    exploration=True,
                )

                action_val = action.item()
                next_state, reward, terminated, truncated, next_info = env.step(
                    action_val
                )
                done = terminated or truncated

                replay_buffer.store(
                    observations=state,
                    actions=action_val,
                    values=float(metadata["value"].item()),
                    old_log_probs=float(metadata["log_prob"].item()),
                    rewards=reward,
                    dones=done,
                    info=info,
                )

                state, info = next_state, next_info
                current_episode_score += reward
                epoch_steps += 1
                steps_collected += 1

                if done or epoch_steps == STEPS_PER_EPOCH:
                    if terminated:
                        last_value = 0.0
                    else:
                        obs_t = torch.tensor(
                            state, dtype=torch.float32, device=DEVICE
                        ).unsqueeze(0)
                        out = agent_network.obs_inference(obs_t)
                        last_value = out.value.item()

                    trajectory_end_index = replay_buffer.size
                    trajectory_slice = slice(
                        trajectory_start_index, trajectory_end_index
                    )

                    if trajectory_end_index > trajectory_start_index:
                        res = replay_buffer.input_processor.finish_trajectory(
                            replay_buffer.buffers,
                            trajectory_slice,
                            last_value=last_value,
                        )
                        if res:
                            for k, v in res.items():
                                replay_buffer.buffers[k][trajectory_slice] = v

                    trajectory_start_index = trajectory_end_index

                    if done:
                        training_scores.append(current_episode_score)
                        avg_training_score = np.mean(training_scores[-100:])
                        if len(training_scores) % 50 == 0:
                            print(
                                f"Game {len(training_scores)} | Score: {current_episode_score} | Avg (L100): {avg_training_score:.2f} | Total Steps: {steps_collected}"
                            )

                        state, info = env.reset()
                        current_episode_score = 0.0

        # Learning Phase
        iterator = PPOEpochIterator(
            replay_buffer=replay_buffer,
            num_epochs=TRAIN_POLICY_ITERATIONS,
            num_minibatches=NUM_MINIBATCHES,
            device=DEVICE,
        )
        for _ in learner.step(iterator):
            pass
        replay_buffer.clear()

        # Early break if solved (475+) to speed up test
        if len(training_scores) >= 100:
            avg_training_score = np.mean(training_scores[-100:])
            if avg_training_score >= 475.0:
                print(
                    f"Solved! Final Avg Training Score (last 100): {avg_training_score:.2f}"
                )
                break

    # --- Assertions ---
    # 1. Average of last 100 training games is 450+
    assert (
        len(training_scores) >= 100
    ), f"Not enough training games completed: {len(training_scores)}"
    avg_training_score = np.mean(training_scores[-100:])
    assert (
        avg_training_score >= 450.0
    ), f"Average training score {avg_training_score:.2f} is below 450.0"

    # 2. Last 3 test scores are 500
    test_scores = evaluate_agent(
        env, agent_network, policy_source, action_selector, DEVICE, num_episodes=3
    )
    print(f"Evaluation scores: {test_scores}")
    for i, score in enumerate(test_scores):
        assert score == 500.0, f"Test episode {i} score {score} is not 500.0"

    env.close()


if __name__ == "__main__":
    test_ppo_cartpole_full_training()
    print("Test passed!")
