import pickle
import torch
import gymnasium as gym
from agents.muzero import MuZeroAgent
from agent_configs.muzero_config import MuZeroConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel


def test_muzero_agent_picklability():
    class DummyGame:
        def __init__(self):
            self.num_players = 1
            self.observation_shape = (1, 8, 8)
            self.num_actions = 2
            self.is_deterministic = True
            self.has_legal_moves = False
            self.reward_threshold = 100
            self.is_discrete = True
            self.is_image = True

        def make_env(self, render_mode=None):
            import gymnasium as gym
            from gymnasium.spaces import Box, Discrete

            class MockSpec:
                def __init__(self):
                    self.reward_threshold = 100
                    self.id = "Mock-v0"
                    self.max_episode_steps = 100

            class MockEnv(gym.Env):
                def __init__(self):
                    self.action_space = Discrete(2)
                    self.observation_space = Box(
                        low=0, high=255, shape=(1, 8, 8), dtype=float
                    )
                    self.num_players = 1
                    self.spec = MockSpec()

                def reset(self, seed=None):
                    return torch.zeros(1, 8, 8), {}

                def step(self, action):
                    return torch.zeros(1, 8, 8), 0.0, False, False, {}

                def close(self):
                    pass

            return MockEnv()

    config_dict = {
        "world_model_cls": MuzeroWorldModel,
        "minibatch_size": 1,
        "training_steps": 100,
        "optimizer": torch.optim.Adam,
        "learning_rate": 0.001,
        "num_simulations": 10,
        "root_dirichlet_alpha": 0.3,
        "root_exploration_fraction": 0.25,
        "pb_c_base": 19652,
        "pb_c_init": 1.25,
        "max_moves": 100,
        "discount_factor": 0.99,
        "td_steps": 5,
        "num_unroll_steps": 5,
        "per_alpha": 0.6,
        "per_beta": 0.4,
        "per_epsilon": 0.01,
        "min_replay_buffer_size": 10,
        "replay_buffer_size": 100,
        "multi_process": False,
        "num_chance": 32,
    }

    game_config = DummyGame()
    config = MuZeroConfig(config_dict, game_config)

    env = game_config.make_env()
    agent = MuZeroAgent(env, config)

    print("Attempting to pickle MuZeroAgent...")
    try:
        import dill

        pickled_agent = dill.dumps(agent)
        print("Agent pickled successfully with dill!")
        unpickled_agent = dill.loads(pickled_agent)
    except ImportError:
        import pickle

        pickled_agent = pickle.dumps(agent)
        print("Agent pickled successfully with pickle!")
        unpickled_agent = pickle.loads(pickled_agent)

    print("Agent unpickled successfully!")

    # Verify GenericActor is recovered and has picklable env_factory
    print(f"Unpickled actor env_factory: {unpickled_agent.actor.env_factory}")

    # Verify we can call the env_factory
    new_env = unpickled_agent.actor.env_factory()
    print("Can successfully call unpickled env_factory!")
    new_env.close()


if __name__ == "__main__":
    test_muzero_agent_picklability()
