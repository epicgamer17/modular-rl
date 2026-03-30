import pytest
import torch
import numpy as np
import random
from agents.environments.adapters import GymAdapter, VectorAdapter, PettingZooAdapter

pytestmark = pytest.mark.unit

@pytest.fixture(autouse=True)
def seed_everything():
    """Ensure strict determinism for every test."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

@pytest.fixture
def device():
    return torch.device("cpu")

class MockGymEnv:
    def __init__(self, action_n=2, obs_shape=(4,)):
        import gymnasium.spaces as spaces
        self.observation_space = spaces.Box(low=0, high=1, shape=obs_shape)
        self.action_space = spaces.Discrete(action_n)
    def reset(self, seed=None, options=None):
        return np.zeros(self.observation_space.shape), {"some": "info"}
    def step(self, action):
        return np.zeros(self.observation_space.shape), 1.0, False, False, {"some": "info"}

def test_gym_adapter_contract(device):
    env = MockGymEnv(action_n=2)
    adapter = GymAdapter(env, device)
    
    obs, info = adapter.reset()
    assert "player_id" in info
    assert "legal_moves_mask" in info
    assert info["player_id"].shape == (1,)
    assert info["legal_moves_mask"].shape == (1, 2)

    action = torch.tensor([0])
    obs, reward, term, trunc, info = adapter.step(action)
    assert "player_id" in info
    assert "legal_moves_mask" in info

class MockVectorEnv:
    def __init__(self, num_envs=4, action_n=3):
        import gymnasium.spaces as spaces
        self.num_envs = num_envs
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_envs, 8))
    def reset(self, seed=None, options=None):
        return np.zeros((self.num_envs, 8)), [{} for _ in range(self.num_envs)]
    def step(self, actions):
        return (
            np.zeros((self.num_envs, 8)), 
            np.zeros(self.num_envs), 
            np.zeros(self.num_envs, dtype=bool), 
            np.zeros(self.num_envs, dtype=bool), 
            [{"legal_moves": [0]} for _ in range(self.num_envs)]
        )

def test_vector_adapter_contract(device):
    num_envs = 4
    num_actions = 3
    env = MockVectorEnv(num_envs=num_envs, action_n=num_actions)
    adapter = VectorAdapter(env, device, num_actions=num_actions)
    
    obs, info = adapter.reset()
    assert obs.shape == (num_envs, 8)
    assert "player_id" in info
    assert "legal_moves_mask" in info
    assert info["player_id"].shape == (num_envs,)

    actions = torch.zeros(num_envs, dtype=torch.long)
    obs, reward, term, trunc, info = adapter.step(actions)
    assert info["player_id"].shape == (num_envs,)
    assert info["legal_moves_mask"].shape == (num_envs, num_actions)

def test_pettingzoo_aec_dict_obs_contract(device):
    """Verifies PettingZooAdapter handles dict observations in AEC mode."""
    class MockAECDictEnv:
        def __init__(self):
            self.possible_agents = ["player_0", "player_1"]
            self.agent_selection = "player_0"
            self.rewards = {"player_0": 0.0, "player_1": 0.0}
        def action_space(self, agent):
            import gymnasium.spaces as spaces
            return spaces.Discrete(5)
        def reset(self, seed=None, options=None):
            pass
        def step(self, action):
            pass
        def last(self):
            # Return dict observation with action_mask
            obs = {
                "observation": np.zeros(10),
                "action_mask": np.array([0, 1, 1, 0, 0]) # 1 and 2 legal
            }
            return obs, 0.0, False, False, {}

    env = MockAECDictEnv()
    adapter = PettingZooAdapter(env, device)
    assert adapter.is_aec
    
    obs, info = adapter.reset()
    assert obs.shape == (1, 10)
    assert "player_id" in info
    assert "legal_moves_mask" in info
    # Verify legal moves extracted from action_mask
    assert info["legal_moves_mask"][0, 1] == True
    assert info["legal_moves_mask"][0, 0] == False
    assert info["legal_moves_mask"][0, 2] == True

def test_pettingzoo_parallel_dict_obs_contract(device):
    """Verifies PettingZooAdapter handles dict observations in Parallel mode."""
    class MockParallelDictEnv:
        def __init__(self):
            self.possible_agents = ["player_0", "player_1"]
            self.agents = ["player_0", "player_1"]
        def action_space(self, agent):
            import gymnasium.spaces as spaces
            return spaces.Discrete(3)
        def reset(self, seed=None, options=None):
            return (
                {
                    "player_0": {"observation": np.zeros(5), "action_mask": np.array([1, 0, 0])},
                    "player_1": {"observation": np.zeros(5), "action_mask": np.array([0, 1, 0])}
                },
                {}
            )
        def step(self, actions):
            return (
                {
                    "player_0": {"observation": np.zeros(5), "action_mask": np.array([1, 0, 0])},
                    "player_1": {"observation": np.zeros(5), "action_mask": np.array([0, 1, 0])}
                },
                {"player_0": 0.0, "player_1": 0.0},
                {"player_0": False, "player_1": False},
                {"player_0": False, "player_1": False},
                {}
            )

    env = MockParallelDictEnv()
    adapter = PettingZooAdapter(env, device)
    
    obs, info = adapter.reset()
    assert obs.shape == (2, 5)
    assert "legal_moves_mask" in info
    assert info["legal_moves_mask"][0, 0] == True
    assert info["legal_moves_mask"][1, 1] == True
    assert info["legal_moves_mask"][0, 1] == False
