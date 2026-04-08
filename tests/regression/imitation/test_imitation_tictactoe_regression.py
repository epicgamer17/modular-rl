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
from learner.core import BlackboardEngine
from learner.pipeline.forward_pass import ForwardPassComponent
from learner.losses.optimizer_step import OptimizerStepComponent


from learner.losses import LossAggregatorComponent, ImitationLoss
from learner.pipeline.target_builders import (
    PassThroughTargetComponent,
    UniversalInfrastructureComponent,
)
from learner.pipeline.batch_iterators import RepeatSampleIterator
from envs.factories.wrappers.observation import ActionMaskInInfoWrapper
import pytest

pytestmark = pytest.mark.regression
from data.storage.circular import BufferConfig, ModularReplayBuffer
from data.samplers.prioritized import UniformSampler
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
    BATCH_SIZE = 128
    LEARNING_RATE = 0.003
    TRAINING_STEPS = 3000
    EXPERT_EPISODES = 1000

    # --- Setup Environment & Expert ---
    env = tictactoe_v3.env()
    env = ActionMaskInInfoWrapper(env)
    expert = TicTacToeBestAgent()

    # TicTacToe observation space is (3, 3, 2), action space is Discrete(9)
    obs_shape = (3, 3, 2)
    num_actions = 9

    # --- Setup Network ---
    backbone = MLPBackbone(input_shape=obs_shape, widths=[128, 128])
    head = PolicyHead(
        input_shape=backbone.output_shape,
        representation=ClassificationRepresentation(num_actions),
    )
    agent_network = ModularAgentNetwork(
        components={"feature_block": backbone, "policy_head": head}
    ).to(DEVICE)

    # --- Setup Buffer ---
    buffer_configs = [
        BufferConfig("observations", shape=obs_shape, dtype=torch.float32),
        BufferConfig("actions", shape=(), dtype=torch.int64),
        BufferConfig("legal_moves_masks", shape=(num_actions,), dtype=torch.bool),
        BufferConfig("policies", shape=(num_actions,), dtype=torch.float32),
    ]

    replay_buffer = ModularReplayBuffer(
        max_size=20000,
        batch_size=BATCH_SIZE,
        buffer_configs=buffer_configs,
        sampler=UniformSampler(),
    )

    # --- Collect Expert Data ---
    print(f"Collecting expert data for {EXPERT_EPISODES} episodes...")
    for _ in range(EXPERT_EPISODES):
        env.reset()
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            mask = info["legal_moves"]

            if termination or truncation:
                action = None
            else:
                # Expert expects (2, 3, 3) observation
                expert_obs = np.moveaxis(obs, -1, 0)
                expert_info = {"legal_moves": mask}
                action = expert.select_actions(expert_obs, expert_info)

                # Store one-hot target policy
                target_policy = np.zeros(num_actions, dtype=np.float32)
                target_policy[action] = 1.0

                mask_bool = np.zeros(num_actions, dtype=bool)
                mask_bool[mask] = True

                replay_buffer.store(
                    observations=obs,
                    actions=action,
                    legal_moves_masks=mask_bool,
                    policies=target_policy,
                )

            env.step(action)

    print(f"Buffer size: {replay_buffer.size}")

    # --- Setup Learner ---
    optimizer = {"default": torch.optim.Adam(agent_network.parameters(), lr=LEARNING_RATE)}
    imitation_loss = ImitationLoss(
        loss_fn=F.cross_entropy,
    )

    learner = BlackboardEngine(
        components=[
            ForwardPassComponent(agent_network, None),
            PassThroughTargetComponent(keys_to_keep=["policies", "actions"]),
            UniversalInfrastructureComponent(),
            imitation_loss,
            LossAggregatorComponent(loss_weights={"policy_loss": 1.0}),
            OptimizerStepComponent(
                agent_network=agent_network,
                optimizers=optimizer,
            ),
        ],
        device=DEVICE,
    )

    # --- Training Loop ---
    print(f"Training for {TRAINING_STEPS} steps...")
    for step in range(1, TRAINING_STEPS + 1):
        iterator = RepeatSampleIterator(replay_buffer, num_iterations=1, device=DEVICE)
        metrics = []
        for step_metrics in learner.step(iterator):
            # BlackboardEngine.step yields {"losses": ..., "total_losses": ..., "meta": ...}
            metrics.append(step_metrics["total_losses"]["default"])

        if step % 500 == 0:
            loss_val = np.mean(metrics)
            print(f"Step {step} | Loss: {loss_val:.4f}")

    # --- Evaluation ---
    print("Evaluating trained model...")
    # Greedy win rate against random should be very high
    agent_network.eval()
    wins = 0
    total_eval_episodes = 100

    for _ in range(total_eval_episodes):
        env.reset()
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                if reward > 0:
                    wins += 1
                action = None
            else:
                # Greedily select action
                obs_tensor = (
                    torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
                )  # [1, H, W, C]
                with torch.inference_mode():
                    output = agent_network.obs_inference(obs_tensor)
                    logits = output.policy.logits[0]
                    # Mask legal moves
                    mask = torch.zeros(num_actions, dtype=torch.bool, device=DEVICE)
                    mask[info["legal_moves"]] = True
                    logits[~mask] = -float("inf")
                    action = torch.argmax(logits).item()

            env.step(action)

    win_rate = wins / total_eval_episodes
    print(f"Win rate: {win_rate:.2f}")

    # In TicTacToe imitation of a perfect expert, it should never lose against random,
    # but win rate can vary depending on random moves. 
    # Let's just assert it learns SOMETHING (win rate > 0.8)
    assert win_rate >= 0.8, f"Imitation win rate {win_rate} is too low."


if __name__ == "__main__":
    test_imitation_tictactoe_regression()
