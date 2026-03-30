import torch
import numpy as np
import pytest
import torch.nn as nn
from agents.workers.actors import RolloutActor
from agents.action_selectors.selectors import ArgmaxSelector
from agents.action_selectors.policy_sources import NetworkPolicySource
from stats.stats import StatTracker
from agents.trainers.base_trainer import BaseTrainer

pytestmark = pytest.mark.unit


class MockNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def obs_inference(self, obs):
        from modules.models.inference_output import InferenceOutput
        from torch.distributions import Categorical

        return InferenceOutput(
            policy=Categorical(logits=torch.ones((1, 2))), value=torch.tensor([0.0])
        )

    def eval(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, d, strict=True):
        pass

    def parameters(self):
        return [nn.Parameter(torch.zeros(1))]

    def to(self, device):
        return self


class MockBuffer:
    def store_aggregate(self, sequence, **kwargs):
        pass


class MockMultiplayerAdapter:
    def __init__(self, device=torch.device("cpu")):
        self.num_players = 2
        self.num_envs = 1
        self.device = device
        self.current_player = 0
        self.steps = 0

    def reset(self, **kwargs):
        self.current_player = 0
        self.steps = 0
        return torch.zeros((1, 4)), {"player_id": [0]}

    def step(self, actions):
        self.steps += 1
        # P1 acts on step 1, P2 acts on step 2.
        # Turn 0: P1 acts, reward 0.
        # Turn 1: P2 acts, reward -10. But P1 gets 10 via all_rewards.
        reward = torch.tensor([0.0], device=self.device)

        all_rewards = {0: 0.0, 1: 0.0}
        if self.steps == 2:
            all_rewards = {0: 10.0, 1: -10.0}
            reward = torch.tensor([-10.0], device=self.device)

        terminated = torch.tensor([self.steps >= 2], device=self.device)
        truncated = torch.tensor([False], device=self.device)

        self.current_player = (self.current_player + 1) % 2
        info = {"player_id": [self.current_player], "all_rewards": all_rewards}
        return (
            torch.zeros((1, 4), device=self.device),
            reward,
            terminated,
            truncated,
            info,
        )


def test_training_score_aec_multi_reward():
    """Verify that RolloutActor captures simultaneous rewards for non-acting players via AEC adapter."""
    net = MockNetwork()
    adapter = MockMultiplayerAdapter()

    actor = RolloutActor(
        adapter_cls=lambda *args, **kwargs: adapter,
        adapter_args=(),
        network=net,
        policy_source=NetworkPolicySource(net),
        buffer=MockBuffer(),
        action_selector=ArgmaxSelector(),
        num_players=2,
    )

    # Run 1 episode (2 steps)
    metrics = actor.collect(num_steps=2)

    assert "batch_scores" in metrics
    assert len(metrics["batch_scores"]) == 1
    scores = metrics["batch_scores"][0]

    # P1 (index 0) should have accumulated 10.0 (from step 2's all_rewards)
    # P2 (index 1) should have accumulated -10.0 (from step 2's reward/all_rewards)
    assert scores[0] == 10.0
    assert scores[1] == -10.0


def test_base_trainer_score_stats_recording():
    """Verify BaseTrainer records score subkeys (p0, p1, avg) correctly."""
    stats = StatTracker("test_stats")

    class DummyConfig:
        def __init__(self):
            self.game = type("Game", (), {"num_players": 2, "num_actions": 2})()
            self.training_steps = 100
            self.multi_process = False
            self.device = "cpu"
            self.test_trials = 1
            self.test_agents = []
            self.test_interval = 10

    res = {"batch_scores": [np.array([25.0, -10.0])], "batch_lengths": [10]}

    env = type(
        "Env",
        (),
        {
            "observation_space": type("OS", (), {"shape": (4,), "dtype": np.float32})(),
            "action_space": type("AS", (), {"n": 2})(),
            "possible_agents": ["p0", "p1"],
            "num_players": 2,
        },
    )()

    trainer = BaseTrainer(DummyConfig(), env, torch.device("cpu"), stats=stats)
    trainer.setup()  # Calls _setup_stats
    trainer._record_collection_metrics([res])

    # Verify subkeys exist and contain correct values
    assert "score" in stats.stats
    data = stats.stats["score"]
    assert "p0" in data, f"Expected 'p0' in score stats, keys: {data.keys()}"
    assert "p1" in data
    assert "avg" in data

    assert data["p0"][0] == 25.0
    assert data["p1"][0] == -10.0
    assert data["avg"][0] == 7.5
