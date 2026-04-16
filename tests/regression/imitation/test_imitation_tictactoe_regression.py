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
from modules.representations import ClassificationRepresentation
from core import BlackboardEngine
from components.neural import ForwardPassComponent
from components.losses import OptimizerStepComponent


from components.losses import LossAggregatorComponent
from components.losses import PolicyLoss
from components.targets import (
    ClassificationFormatterComponent,
)
from core import RepeatSampleIterator
from envs.factories.wrappers.observation import ActionMaskInInfoWrapper
import pytest
from utils.plotting import plot_regression_results


pytestmark = pytest.mark.regression
from data.storage.circular import BufferConfig, ModularReplayBuffer
from data.samplers.prioritized import UniformSampler
from components.experts.tictactoe import TicTacToeBestAgent
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
    TRAINING_STEPS = 30000
    EXPERT_EPISODES = 10000

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
    optimizer = {
        "default": torch.optim.Adam(agent_network.parameters(), lr=LEARNING_RATE)
    }
    imitation_loss = PolicyLoss(
        loss_fn=F.cross_entropy,
        target_key="data.policies",
    )

    from core.contracts import (
        Key,
        Observation,
        Action,
        SemanticType,
        Policy,
        Mask,
        Probs,
        ShapeContract,
    )

    initial_keys = {
        Key("data.observations", Observation),
        Key("data.actions", Action),
        Key("data.legal_moves_masks", Mask),
        Key(
            "data.policies",
            Policy[Probs],
            shape=ShapeContract(
                time_dim=1, symbolic=("B", "T"), dtype=torch.float32, ndim=2
            ),
        ),
    }

    learner = BlackboardEngine(
        components=[
            ForwardPassComponent(agent_network),
            imitation_loss,
            LossAggregatorComponent(loss_weights={"policy_loss": 1.0}),
            OptimizerStepComponent(
                agent_network=agent_network,
                optimizers=optimizer,
            ),
        ],
        initial_keys=initial_keys,
        device=DEVICE,
        strict=True,
    )

    # --- Training Loop ---
    print(f"Training for {TRAINING_STEPS} steps...")
    total_losses = []
    for step in range(1, TRAINING_STEPS + 1):
        iterator = RepeatSampleIterator(replay_buffer, num_iterations=1, device=DEVICE)
        metrics = []
        for step_metrics in learner.step(iterator):
            # BlackboardEngine.step yields {"losses": ..., "total_losses": ..., "meta": ...}
            loss_val = step_metrics["total_losses"]["default"]
            metrics.append(loss_val)
            total_losses.append(loss_val)

        if step % 500 == 0:
            loss_val = np.mean(metrics)
            print(f"Step {step} | Loss: {loss_val:.4f}")

    # --- Evaluation ---
    print("Evaluating trained model against Random...")
    agent_network.eval()

    def run_eval(opponent_type="random", num_episodes=100):
        wins = 0
        draws = 0
        losses = 0
        for _ in range(num_episodes):
            env.reset()
            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()

                if agent == "player_1":
                    if termination or truncation:
                        if reward > 0:
                            wins += 1
                        elif reward < 0:
                            losses += 1
                        else:
                            draws += 1
                        action = None
                    else:
                        obs_tensor = (
                            torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
                        )
                        with torch.inference_mode():
                            output = agent_network.obs_inference(obs_tensor)
                            logits = output.policy.logits[0]
                            mask = torch.zeros(
                                num_actions, dtype=torch.bool, device=DEVICE
                            )
                            mask[info["legal_moves"]] = True
                            logits[~mask] = -float("inf")
                            action = torch.argmax(logits).item()
                else:  # player_2 (the opponent)
                    if termination or truncation:
                        action = None
                    else:
                        mask = info["legal_moves"]
                        if opponent_type == "random":
                            action = int(random.choice(mask))
                        else:  # expert
                            expert_obs = np.moveaxis(obs, -1, 0)
                            expert_info = {"legal_moves": mask}
                            action = expert.select_actions(expert_obs, expert_info)
                env.step(action)
        return wins, draws, losses

    # 1. Eval against Random
    rand_wins, rand_draws, rand_losses = run_eval("random", 100)
    win_rate = rand_wins / 100
    print(
        f"Against Random | Wins: {rand_wins}, Draws: {rand_draws}, Losses: {rand_losses}"
    )

    # 2. Eval against Expert
    exp_wins, exp_draws, exp_losses = run_eval("expert", 100)
    print(
        f"Against Expert | Wins: {exp_wins}, Draws: {exp_draws}, Losses: {exp_losses}"
    )

    # In TicTacToe imitation of a perfect expert, it should never lose against random.
    # We assert it wins or draws almost all games.
    assert rand_losses <= 1, f"Imitation agent lost {rand_losses} games against RANDOM."
    assert win_rate >= 0.7, f"Imitation win rate {win_rate} against random is too low."

    # Against expert, it should at least draw most games if it learned correctly.
    # A perfect player never loses Tic-Tac-Toe.
    assert exp_losses <= 5, f"Imitation agent lost {exp_losses} games against EXPERT."

    # Plot results (Note: No training scores as this is supervised learning)
    plot_regression_results(
        name="Imitation TicTacToe",
        train_scores=[],
        test_scores=[win_rate],
        losses={"Policy Loss": total_losses},
    )


if __name__ == "__main__":
    test_imitation_tictactoe_regression()
