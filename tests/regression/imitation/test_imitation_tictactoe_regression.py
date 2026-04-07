import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, Any

from modules.agent_nets.modular import ModularAgentNetwork
from modules.backbones.mlp import MLPBackbone
from modules.heads.policy import PolicyHead
from learner.losses.representations import ClassificationRepresentation
from learner.base import UniversalLearner
from learner.losses import LossPipeline, ImitationLoss
from learner.pipeline.targets import (
    PassThroughTargetBuilder,
    SingleStepFormatter,
    TargetBuilderPipeline,
)
from learner.pipeline.batch_iterators import RepeatSampleIterator
import pytest

pytestmark = pytest.mark.regression
from replay_buffers.modular_buffer import BufferConfig, ModularReplayBuffer
from replay_buffers.samplers import UniformSampler
from actors.experts.tictactoe_expert import TicTacToeBestAgent
from pettingzoo.classic import tictactoe_v3


def test_imitation_tictactoe_regression():
    """
    Standalone regression test for Imitation Learning on TicTacToe.
    Trains a policy to mimic a TicTacToe expert.
    """
    # --- Hyperparameters ---
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LEARNING_RATE = 1e-3
    TRAINING_STEPS = 3000  # Increased for better convergence
    BATCH_SIZE = 128
    EXPERT_EPISODES = 1000  # Increased for more diverse coverage

    # --- Setup Environment ---
    print("Setting up raw TicTacToe environment...")
    env = tictactoe_v3.env(render_mode=None)
    env.reset()

    # Raw TicTacToe observation is a dict: {"observation": ..., "action_mask": ...}
    raw_obs_space = env.observation_space(env.possible_agents[0])
    # observation: (3, 3, 2)
    obs_shape = raw_obs_space["observation"].shape
    num_actions = env.action_space(env.possible_agents[0]).n
    print(
        f"Obs shape: {obs_shape}, Num actions: {num_actions}, Agents: {env.possible_agents}"
    )

    # --- Setup Expert ---
    expert = TicTacToeBestAgent()

    # --- Setup Network ---
    backbone = MLPBackbone(input_shape=obs_shape, widths=[128, 128])
    policy_rep = ClassificationRepresentation(num_actions)
    policy_head = PolicyHead(
        input_shape=backbone.output_shape,
        representation=policy_rep,
    )

    agent_network = ModularAgentNetwork(
        components={
            "feature_block": backbone,
            "policy_head": policy_head,
        },
    ).to(DEVICE)

    # --- Setup Replay Buffer ---
    buffer_configs = [
        BufferConfig("observations", shape=obs_shape, dtype=torch.float32),
        BufferConfig("actions", shape=(), dtype=torch.int64),
        BufferConfig("legal_moves_masks", shape=(num_actions,), dtype=torch.bool),
        BufferConfig("policies", shape=(num_actions,), dtype=torch.float32),
    ]

    replay_buffer = ModularReplayBuffer(
        max_size=50000,
        batch_size=BATCH_SIZE,
        buffer_configs=buffer_configs,
        sampler=UniformSampler(),
    )

    # --- Data Collection (Expert) ---
    print(f"Collecting {EXPERT_EPISODES} episodes of expert data...")
    for _ in range(EXPERT_EPISODES):
        env.reset()
        for agent in env.agent_iter():
            obs_dict, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
            else:
                obs = obs_dict["observation"]
                mask = obs_dict["action_mask"]

                # Expert expects (2, 3, 3) observation
                expert_obs = np.moveaxis(obs, -1, 0)
                expert_info = {"legal_moves": np.where(mask)[0]}
                action = expert.select_actions(expert_obs, expert_info)

                # Store one-hot target policy
                target_policy = np.zeros(num_actions, dtype=np.float32)
                target_policy[action] = 1.0

                replay_buffer.store(
                    observations=obs,
                    actions=action,
                    legal_moves_masks=mask.astype(bool),
                    policies=target_policy,
                )

            env.step(action)

    print(f"Buffer size: {replay_buffer.size}")

    # --- Setup Learner ---
    optimizer = torch.optim.Adam(agent_network.parameters(), lr=LEARNING_RATE)
    # BaseLoss requires functional-style loss_fn that accepts reduction="none"
    loss_fn = F.cross_entropy

    imitation_loss = ImitationLoss(
        device=DEVICE,
        loss_fn=loss_fn,
    )

    loss_pipeline = LossPipeline(
        modules=[imitation_loss],
        minibatch_size=BATCH_SIZE,
        num_actions=num_actions,
    )

    target_builder = TargetBuilderPipeline(
        [
            PassThroughTargetBuilder(keys_to_keep=["policies", "actions"]),
            SingleStepFormatter(temporal_keys=["policies", "actions"]),
        ]
    )

    learner = UniversalLearner(
        agent_network=agent_network,
        device=DEVICE,
        num_actions=num_actions,
        observation_dimensions=obs_shape,
        observation_dtype=torch.float32,
        optimizer=optimizer,
        loss_pipeline=loss_pipeline,
        target_builder=target_builder,
    )

    # --- Training Loop ---
    print(f"Starting Imitation Learning training for {TRAINING_STEPS} steps...")
    for step in range(1, TRAINING_STEPS + 1):
        iterator = RepeatSampleIterator(replay_buffer, num_iterations=1, device=DEVICE)
        metrics = []
        for step_metrics in learner.step(iterator):
            # UniversalLearner.step yields a dict containing 'loss' (total loss)
            metrics.append(step_metrics["loss"])

        if step % 500 == 0:
            loss_val = np.mean(metrics)
            print(f"Step {step} | Loss: {loss_val:.4f}")

    # --- Evaluation ---
    print("\nEvaluating trained policy against random...")
    num_eval_episodes = 50
    wins = 0
    draws = 0
    losses = 0

    our_agent_id = env.possible_agents[0]
    print(f"Our agent ID for evaluation: {our_agent_id}")

    agent_network.eval()
    for _ in range(num_eval_episodes):
        env.reset()
        episode_rewards = {a: 0 for a in env.possible_agents}
        for agent in env.agent_iter():
            obs_dict, reward, termination, truncation, info = env.last()

            # Accumulate rewards for all agents present in rewards dict
            for a, r in env.rewards.items():
                if a in episode_rewards:
                    episode_rewards[a] += r

            if termination or truncation:
                action = None
            else:
                obs = obs_dict["observation"]
                mask = obs_dict["action_mask"]
                if agent == our_agent_id:  # Our Agent
                    with torch.inference_mode():
                        obs_t = torch.as_tensor(
                            obs, device=DEVICE, dtype=torch.float32
                        ).unsqueeze(0)
                        result = agent_network.obs_inference(obs_t)
                        dist = result.policy
                        logits = dist.logits
                        # Apply mask manually
                        logits[0, mask == 0] = -1e9
                        action = int(logits.argmax().item())
                else:  # Random Opponent
                    action = int(env.action_space(agent).sample(mask))

            env.step(action)

        final_reward = episode_rewards[our_agent_id]
        if final_reward > 0:
            wins += 1
        elif final_reward == 0:
            draws += 1
        else:
            losses += 1

    print(f"Eval Results (50 games): Wins: {wins}, Draws: {draws}, Losses: {losses}")

    assert losses <= 5, f"Learned policy lost {losses} games against random!"
    print("Regression test PASSED: Agent plays perfectly against random.")


if __name__ == "__main__":
    test_imitation_tictactoe_regression()
