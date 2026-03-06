import pytest
pytestmark = [pytest.mark.integration, pytest.mark.slow]

import torch
import numpy as np
from agents.workers.puffer_actor import GymPufferActor
from replay_buffers.sequence import Sequence
from dataclasses import dataclass


@dataclass
class MockConfig:
    num_envs_per_worker: int = 2
    num_players: int = 1
    num_puffer_workers: int = 2


class MockNetwork:
    def __init__(self):
        self.input_shape = (4,)

    def obs_inference(self, obs):
        from types import SimpleNamespace

        batch_size = obs.shape[0] if obs.dim() > 1 else 1
        return SimpleNamespace(
            policy=torch.zeros((batch_size, 2)), value=torch.zeros((batch_size,))
        )

    def to(self, device):
        return self


class MockSelector:
    def select_action(self, agent_network, obs, info, exploration=True, **kwargs):
        batch_size = obs.shape[0] if obs.dim() > 1 else 1
        return torch.zeros(batch_size, dtype=torch.long), {
            "policy": torch.zeros((batch_size, 2)),
            "value": torch.zeros((batch_size,)),
        }


class MockBuffer:
    def __init__(self):
        self.stored_sequences = []

    def store_aggregate(self, sequence):
        self.stored_sequences.append(sequence)


def make_mock_env():
    import gymnasium as gym

    class TinyEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(0, 1, (4,))
            self.action_space = gym.spaces.Discrete(2)
            self.steps = 0

        def reset(self, seed=None, options=None):
            self.steps = 0
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            self.steps += 1
            done = self.steps >= 3
            return np.zeros(4, dtype=np.float32), 1.0, done, False, {}

    return TinyEnv()


def test_puffer_sequence_length():
    config = MockConfig()
    net = MockNetwork()
    sel = MockSelector()
    buf = MockBuffer()

    actor = GymPufferActor(
        env_factory=make_mock_env,
        agent_network=net,
        action_selector=sel,
        replay_buffer=buf,
        num_players=1,
        config=config,
    )

    # Run one episode step by step
    print("Running PufferActor episodes...")
    actor.play_sequence()

    assert len(buf.stored_sequences) > 0, "No sequences stored"
    seq = buf.stored_sequences[0]

    n_obs = len(seq.observation_history)
    n_actions = len(seq.action_history)
    n_rewards = len(seq.rewards)

    print(f"Sequence stats: {n_obs} obs, {n_actions} actions, {n_rewards} rewards")

    # Requirement: len(obs) == len(actions) + 1
    assert (
        n_obs == n_actions + 1
    ), f"Sequence length mismatch: {n_obs} obs, {n_actions} actions"
    assert (
        n_rewards == n_actions
    ), f"Reward length mismatch: {n_rewards} rewards, {n_actions} actions"

    print("Verification SUCCESS: PufferActor sequence length is correct.")


if __name__ == "__main__":
    test_puffer_sequence_length()
