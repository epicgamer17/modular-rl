import pytest
import torch
import numpy as np
import random
import gymnasium as gym
try:
    import pettingzoo.classic.tictactoe_v3 as tictactoe_v3
    from pettingzoo.classic import rps_v2
    HAS_PETTINGZOO = True
except ImportError:
    HAS_PETTINGZOO = False

from agents.environments.adapters import GymAdapter, VectorAdapter, PettingZooAdapter

pytestmark = pytest.mark.integration

@pytest.fixture(autouse=True)
def seed_everything():
    """Ensure strict determinism for every test."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

@pytest.fixture
def device():
    return torch.device("cpu")

def test_gym_adapter_example(device):
    """Verifies GymAdapter with real CartPole environment."""
    env_factory = lambda: gym.make("CartPole-v1")
    adapter = GymAdapter(env_factory, device)
    
    obs, info = adapter.reset()
    assert "player_id" in info
    assert "legal_moves_mask" in info
    assert info["player_id"].shape == (1,)
    assert info["legal_moves_mask"].dim() == 2

    # Step
    actions = torch.zeros(1, dtype=torch.long, device=device)
    obs, reward, term, trunc, info = adapter.step(actions)
    assert "player_id" in info
    assert "legal_moves_mask" in info

@pytest.mark.skipif(not HAS_PETTINGZOO, reason="PettingZoo not installed")
def test_pettingzoo_aec_example(device):
    """Verifies PettingZooAdapter with real TicTacToe AEC environment."""
    env_factory = lambda: tictactoe_v3.env()
    adapter = PettingZooAdapter(env_factory, device)
    assert adapter.is_aec

    obs, info = adapter.reset()
    assert "player_id" in info
    assert "legal_moves_mask" in info
    assert info["player_id"].item() == 0  # player_1 (indexed as 0) starts
    assert info["legal_moves_mask"].shape == (1, 9)

    # Step
    action = torch.tensor([0], device=device)
    obs, reward, term, trunc, info = adapter.step(action)
    assert info["player_id"].item() == 1  # Now player_2 (indexed as 1)
    assert "legal_moves_mask" in info

@pytest.mark.skipif(not HAS_PETTINGZOO, reason="PettingZoo not installed")
def test_pettingzoo_parallel_example(device):
    """Verifies PettingZooAdapter with real RPS Parallel environment."""
    env_factory = lambda: rps_v2.parallel_env()
    adapter = PettingZooAdapter(env_factory, device)
    assert not adapter.is_aec

    obs, info = adapter.reset()
    assert "player_id" in info
    assert torch.equal(info["player_id"].cpu(), torch.tensor([0, 1]))
    assert "legal_moves_mask" in info
    assert info["legal_moves_mask"].shape == (2, 3)

    # Step
    actions = torch.zeros(2, dtype=torch.long, device=device)
    obs, reward, term, trunc, info = adapter.step(actions)
    assert "player_id" in info

def test_vector_adapter_example(device):
    """Verifies VectorAdapter with real Gymnasium SyncVectorEnv."""
    num_envs = 2
    def make_env(): return gym.make("CartPole-v1")
    vec_env = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
    adapter = VectorAdapter(vec_env, device, num_actions=2)

    obs, info = adapter.reset()
    assert obs.shape == (num_envs, 4)
    assert "player_id" in info
    assert "legal_moves_mask" in info
    assert info["player_id"].shape == (num_envs,)

    # Step
    actions = torch.zeros(num_envs, dtype=torch.long, device=device)
    obs, reward, term, trunc, info = adapter.step(actions)
    assert info["player_id"].shape == (num_envs,)
    assert info["legal_moves_mask"].shape == (num_envs, 2)
