import sys
import os
import torch
import numpy as np
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.agent import MARLBaseAgent, BaseAgent
from configs.base import Config


def test_marl_test_fix():
    print("Testing MARLBaseAgent.test fix...")

    # Mock Config
    config = MagicMock(spec=Config)
    config.game = MagicMock()
    config.game.num_players = 2
    config.training_steps = 100
    config.soft_update = False

    # Mock Environment
    env = MagicMock()
    env.possible_agents = ["player_1", "player_2"]
    env.agents = ["player_1", "player_2"]
    env.agent_selection = "player_1"
    env.rewards = {"player_1": 1.0, "player_2": -1.0}
    env.render_mode = None
    env.observation_space = MagicMock()
    env.observation_space.shape = (4,)
    env.observation_space.dtype = np.float32
    env.action_space = MagicMock()
    env.action_space.n = 2

    # Mock last() return value: state, reward, terminated, truncated, info
    # First call (init): not done
    # Second call (after step): done
    env.last.side_effect = [
        (np.zeros((4,)), 0.0, False, False, {}),
        (np.zeros((4,)), 0.0, True, False, {}),
    ]

    # Deepcopy might fail on mocks, so we mock make_test_env
    # Or just use the env as is

    # Instantiate Agent
    # We'll monkeypatch make_test_env to avoid deepcopy issues with mocks
    original_make_test_env = BaseAgent.make_test_env
    BaseAgent.make_test_env = lambda self, e: e

    class DummyMARLAgent(MARLBaseAgent):
        def predict(self, state, info, env=None):
            return torch.tensor([0])

        def select_actions(self, prediction, info):
            return torch.tensor([0])

        def train(self):
            pass

    try:
        agent = DummyMARLAgent(env, config, "test_agent")

        # Run test with player=0 (which maps to player_1)
        # It should try to access env.rewards["player_1"] (via possible_agents[0])
        # Instead of env.rewards["player_0"]
        results = agent.test(num_trials=1, player=0)
        print("Success! Results:", results)
        assert results["score"] == 1.0

    except KeyError as e:
        print(f"Failed! Still getting KeyError: {e}")
        # Restore original for clean exit logic if needed, but we're exiting anyway
        BaseAgent.make_test_env = original_make_test_env
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        BaseAgent.make_test_env = original_make_test_env
        sys.exit(1)
    finally:
        BaseAgent.make_test_env = original_make_test_env


if __name__ == "__main__":
    test_marl_test_fix()
