import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, Any, List, Optional, Iterator, Generator

from modules.agent_nets.modular import ModularAgentNetwork
from modules.backbones.mlp import MLPBackbone
from modules.heads.policy import PolicyHead
from modules.heads.q import QHead
from learner.losses.representations import (
    ClassificationRepresentation,
    ScalarRepresentation,
)
from learner.core import UniversalLearner
from learner.pipeline.base import DeviceTransferComponent
from learner.pipeline.forward_pass import ForwardPassComponent
from learner.losses.optimizer_step import OptimizerStepComponent
from learner.pipeline.wrappers import TargetBuilderComponent, ComponentCallbacks

from learner.losses import LossAggregator, ImitationLoss, QBootstrappingLoss
from learner.pipeline.target_builders import (
    PassThroughTargetBuilder,
    SingleStepFormatter,
    TemporalDifferenceBuilder,
    TargetBuilderPipeline,
)
from learner.pipeline.batch_iterators import RepeatSampleIterator
import pytest

pytestmark = pytest.mark.regression
from data.storage.circular import BufferConfig, ModularReplayBuffer
from data.samplers.prioritized import UniformSampler
from pettingzoo.classic import leduc_holdem_v4


class ReservoirBuffer:
    """A simple Reservoir Buffer for NFSP's supervised learning part."""

    def __init__(
        self, max_size: int, obs_shape: tuple, num_actions: int, device: torch.device
    ):
        self.max_size = max_size
        self.device = device
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.observations = torch.zeros((max_size, *obs_shape), dtype=torch.float32)
        self.actions = torch.zeros((max_size,), dtype=torch.int64)
        self.count = 0
        self.current_size = 0

    def store(self, obs, action):
        if self.current_size < self.max_size:
            self.observations[self.current_size] = torch.as_tensor(obs)
            self.actions[self.current_size] = torch.as_tensor(action)
            self.current_size += 1
        else:
            # Reservoir sampling logic
            idx = random.randint(0, self.count)
            if idx < self.max_size:
                self.observations[idx] = torch.as_tensor(obs)
                self.actions[idx] = torch.as_tensor(action)
        self.count += 1

    def sample(self, batch_size: int):
        if self.current_size < batch_size:
            return None
        indices = np.random.choice(self.current_size, batch_size, replace=False)

        # UniversalLearner expects [B, ...] for single-step logic in learner_inference
        obs = self.observations[indices].to(self.device)
        actions = self.actions[indices].to(self.device)

        # ImitationLoss expects 'policies' key to match the head's output shape [B, T, num_actions] (after T=1 unsqueeze)
        # We provide [B, num_actions] here, and SingleStepFormatter will make it [B, 1, num_actions]
        one_hot_policies = (
            F.one_hot(self.actions[indices], num_classes=self.num_actions)
            .float()
            .to(self.device)
        )

        return {
            "observations": obs,
            "actions": actions,
            "policies": one_hot_policies,
        }


def test_nfsp_leduc_regression():
    """
    Standalone regression test for Neural Fictitious Self-Play (NFSP) on Leduc Hold'em.
    Tests if NFSP can learn a reasonable strategy in a simple poker game.
    """
    # --- Hyperparameters ---
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LR_RL = 1e-2
    LR_SL = 1e-2
    TRAINING_EPISODES = 10000
    BATCH_SIZE = 128
    ANTICIPATORY_PARAM = 0.1  # eta
    EPSILON_START = 0.06
    EPSILON_END = 0.01
    EPSILON_DECAY = 4000

    # --- Setup Environment ---
    print("Setting up Leduc Hold'em environment...")
    env = leduc_holdem_v4.env(render_mode=None)
    env.reset()

    raw_obs_space = env.observation_space(env.possible_agents[0])
    obs_shape = raw_obs_space["observation"].shape
    num_actions = env.action_space(env.possible_agents[0]).n
    print(f"Obs shape: {obs_shape}, Num actions: {num_actions}")

    # --- Setup Networks ---
    def make_rl_net():
        backbone = MLPBackbone(input_shape=obs_shape, widths=[128, 128])
        # QHead needs representation and hidden_widths
        head = QHead(
            input_shape=backbone.output_shape,
            representation=ScalarRepresentation(),
            hidden_widths=[],
            num_actions=num_actions,
        )
        return ModularAgentNetwork(
            components={"feature_block": backbone, "q_head": head}
        ).to(DEVICE)

    rl_network = make_rl_net()
    rl_target_network = make_rl_net()
    rl_target_network.load_state_dict(rl_network.state_dict())

    # 2. Average Policy (SL)
    sl_backbone = MLPBackbone(input_shape=obs_shape, widths=[128, 128])
    # PolicyHead needs representation
    sl_head = PolicyHead(
        input_shape=sl_backbone.output_shape,
        representation=ClassificationRepresentation(num_actions),
    )
    sl_network = ModularAgentNetwork(
        components={"feature_block": sl_backbone, "policy_head": sl_head}
    ).to(DEVICE)

    # --- Setup Learners ---
    # 1. RL Learner
    rl_optimizer = torch.optim.Adam(rl_network.parameters(), lr=LR_RL)
    rl_loss = QBootstrappingLoss(device=DEVICE)
    rl_loss_pipeline = LossAggregator(
        modules=[rl_loss], minibatch_size=BATCH_SIZE, num_actions=num_actions
    )
    rl_target_builder = TargetBuilderPipeline(
        [
            TemporalDifferenceBuilder(
                target_network=rl_target_network, gamma=0.99, n_step=1
            ),
            SingleStepFormatter(),
        ]
    )
    rl_learner = UniversalLearner(
        components=[
            DeviceTransferComponent(DEVICE),
            ForwardPassComponent(rl_network, None),
            TargetBuilderComponent(rl_target_builder, rl_network),
            rl_loss_pipeline,
            OptimizerStepComponent(
                agent_network=rl_network,
                optimizer=rl_optimizer,
            ),
        ]
    )

    # 2. SL Learner
    sl_optimizer = torch.optim.Adam(sl_network.parameters(), lr=LR_SL)
    sl_loss = ImitationLoss(
        device=DEVICE, loss_fn=F.cross_entropy, target_key="actions"
    )
    sl_loss_pipeline = LossAggregator(
        modules=[sl_loss], minibatch_size=BATCH_SIZE, num_actions=num_actions
    )
    sl_target_builder = TargetBuilderPipeline(
        [
            PassThroughTargetBuilder(keys_to_keep=["policies", "actions"]),
            SingleStepFormatter(),  # Use default to cover policies and actions
        ]
    )
    sl_learner = UniversalLearner(
        components=[
            DeviceTransferComponent(DEVICE),
            ForwardPassComponent(sl_network, None),
            TargetBuilderComponent(sl_target_builder, sl_network),
            sl_loss_pipeline,
            OptimizerStepComponent(
                agent_network=sl_network,
                optimizer=sl_optimizer,
            ),
        ]
    )

    # --- Setup Buffers ---
    rl_buffer = ModularReplayBuffer(
        max_size=20000,
        batch_size=BATCH_SIZE,
        buffer_configs=[
            BufferConfig("observations", shape=obs_shape, dtype=torch.float32),
            BufferConfig("actions", shape=(), dtype=torch.int64),
            BufferConfig("rewards", shape=(), dtype=torch.float32),
            BufferConfig("dones", shape=(), dtype=torch.bool),
            BufferConfig("next_observations", shape=obs_shape, dtype=torch.float32),
            BufferConfig(
                "next_legal_moves_masks",
                shape=(num_actions,),
                dtype=torch.bool,
                fill_value=True,
            ),
        ],
        sampler=UniformSampler(),
    )
    sl_buffer = ReservoirBuffer(
        max_size=50000, obs_shape=obs_shape, num_actions=num_actions, device=DEVICE
    )

    # --- NFSP Training Loop ---
    print(f"Starting NFSP training for {TRAINING_EPISODES} episodes...")

    total_steps = 0
    for episode in range(1, TRAINING_EPISODES + 1):
        env.reset()

        last_obs = {a: None for a in env.possible_agents}
        last_action = {a: None for a in env.possible_agents}

        for agent in env.agent_iter():
            obs_dict, reward, termination, truncation, info = env.last()

            # Store transitions for RL buffer
            if last_obs[agent] is not None:
                rl_buffer.store(
                    observations=last_obs[agent],
                    actions=last_action[agent],
                    rewards=reward,
                    dones=termination or truncation,
                    next_observations=obs_dict["observation"],
                    next_legal_moves_masks=obs_dict["action_mask"],
                )

            if termination or truncation:
                action = None
            else:
                obs = obs_dict["observation"]
                mask = obs_dict["action_mask"]

                # NFSP Strategy Selection
                if random.random() < ANTICIPATORY_PARAM:
                    # Act optimally (DQN) - Best Response
                    eps = max(EPSILON_END, EPSILON_START - (episode / EPSILON_DECAY))
                    if random.random() < eps:
                        action = int(env.action_space(agent).sample(mask))
                    else:
                        with torch.no_grad():
                            obs_t = torch.as_tensor(
                                obs, device=DEVICE, dtype=torch.float32
                            ).unsqueeze(0)
                            q_vals = rl_network.obs_inference(obs_t).q_values
                            q_vals[0, mask == 0] = -1e9
                            action = int(q_vals.argmax().item())

                    # Store (s, a) in SL buffer if we acted according to Best Response
                    sl_buffer.store(obs, action)
                else:
                    # Act according to Average Policy (Supervised)
                    with torch.no_grad():
                        obs_t = torch.as_tensor(
                            obs, device=DEVICE, dtype=torch.float32
                        ).unsqueeze(0)
                        dist = sl_network.obs_inference(obs_t).policy
                        logits = dist.logits
                        logits[0, mask == 0] = -1e9
                        action = int(logits.argmax().item())

                last_obs[agent] = obs
                last_action[agent] = action

            env.step(action)
            total_steps += 1

        # Perform learning steps
        if episode > 100:
            # RL Update
            if rl_buffer.size >= BATCH_SIZE:
                rl_iterator = RepeatSampleIterator(
                    rl_buffer, num_iterations=1, device=DEVICE
                )
                # UniversalLearner.step returns an iterator of results
                for _ in rl_learner.step(rl_iterator):
                    pass

            # SL Update
            sl_batch = sl_buffer.sample(BATCH_SIZE)
            if sl_batch is not None:

                # UniversalLearner needs an iterator that yields dicts
                class SimpleIterator:
                    def __init__(self, data):
                        self.data = data
                        self.done = False

                    def __next__(self):
                        if self.done:
                            raise StopIteration
                        self.done = True
                        return self.data

                    def __iter__(self):
                        return self

                for _ in sl_learner.step(SimpleIterator(sl_batch)):
                    pass

        if episode % 1000 == 0:
            print(f"Episode {episode} completed. Total steps: {total_steps}")
            # Target network sync
            rl_target_network.load_state_dict(rl_network.state_dict())

    # --- Evaluation ---
    print("\nEvaluating NFSP Average Policy against Random...")
    num_eval_games = 100
    total_reward = 0

    sl_network.eval()
    for _ in range(num_eval_games):
        env.reset()
        episode_reward = 0
        for agent in env.agent_iter():
            obs_dict, reward, termination, truncation, info = env.last()

            if agent == "player_0":
                episode_reward += reward

            if termination or truncation:
                action = None
            else:
                obs = obs_dict["observation"]
                mask = obs_dict["action_mask"]
                with torch.no_grad():
                    obs_t = torch.as_tensor(
                        obs, device=DEVICE, dtype=torch.float32
                    ).unsqueeze(0)
                    dist = sl_network.obs_inference(obs_t).policy
                    logits = dist.logits
                    logits[0, mask == 0] = -1e9
                    action = int(logits.argmax().item())
            env.step(action)
        total_reward += episode_reward

    avg_reward = total_reward / num_eval_games
    print(f"Average Reward (Player 0) against Random: {avg_reward:.3f}")

    # In Leduc, a basic strategy should always beat random (avg reward > 0)
    assert avg_reward > 0.0, f"NFSP failed! Avg reward: {avg_reward}"
    print("Regression test PASSED: NFSP policy evaluated successfully.")


if __name__ == "__main__":
    test_nfsp_leduc_regression()
