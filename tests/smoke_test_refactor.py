import torch
import numpy as np
import gym
from agents.muzero_policy import MuZeroPolicy
from agents.actors import GenericActor
from agent_configs.muzero_config import MuZeroConfig


def test_generic_actor_smoke():
    class DummyGame:
        def __init__(self):
            self.num_players = 1
            self.observation_shape = (4,)
            self.num_actions = 2
            self.is_deterministic = True
            self.has_legal_moves = False

        def make_env(self):
            return gym.make("CartPole-v1")

    # MuZeroConfig expects a dict and game_config
    config_dict = {
        "world_model_cls": "dummy",
        "minibatch_size": 1,
        "training_steps": 100,
        "optimizer": "adam",
        "lr": 0.001,
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
    }

    game_config = DummyGame()
    config = MuZeroConfig(config_dict, game_config)
    config.temperatures = [1.0]
    config.temperature_updates = []

    device = torch.device("cpu")
    num_actions = 2
    obs_dims = (4,)

    policy = MuZeroPolicy(config, device, num_actions, obs_dims)

    # Mocking search to avoid full MCTS in smoke test
    class MockSearch:
        def run(self, *args, **kwargs):
            # Return valid uniform probabilities
            policy_logits = torch.ones(num_actions) / num_actions
            return (torch.zeros(1), policy_logits, policy_logits, torch.tensor(0), {})

    policy.search = MockSearch()

    actor = GenericActor(config.game.make_env, policy)

    print("Running episode...")
    game = actor.play_game()
    print(f"Episode finished. Length: {len(game)}")
    assert len(game) > 0
    print("Smoke test passed!")


if __name__ == "__main__":
    test_generic_actor_smoke()
