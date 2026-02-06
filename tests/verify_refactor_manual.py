import torch
import torch.multiprocessing as mp
from unittest.mock import MagicMock


# Mock multiprocessing to avoid shared memory issues on this environment
class MockValue:
    def __init__(self, typecode, value):
        self.value = value


mp.Value = MockValue
mp.Queue = MagicMock

from torch.optim import Adam
import gymnasium as gym
from agents.muzero import MuZeroAgent
from agent_configs.muzero_config import MuZeroConfig
from game_configs.cartpole_config import CartPoleConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from losses.basic_losses import CategoricalCrossentropyLoss


def action_as_onehot(action, num_actions):
    if isinstance(action, torch.Tensor):
        action = action.item()
    one_hot = torch.zeros(num_actions)
    one_hot[action] = 1.0
    return one_hot


def verify_agent_refactor():
    print("Initializing environment...")
    game_config = CartPoleConfig()
    env = game_config.make_env()

    config_dict = {
        "world_model_cls": MuzeroWorldModel,
        "residual_layers": [],
        "representation_dense_layer_widths": [64],
        "dynamics_dense_layer_widths": [64],
        "actor_dense_layer_widths": [64],
        "critic_dense_layer_widths": [64],
        "reward_dense_layer_widths": [64],
        "actor_conv_layers": [],
        "critic_conv_layers": [],
        "reward_conv_layers": [],
        "to_play_conv_layers": [],
        "num_simulations": 2,
        "minibatch_size": 2,
        "min_replay_buffer_size": 2,
        "replay_buffer_size": 10,
        "unroll_steps": 2,
        "n_step": 2,
        "multi_process": False,
        "training_steps": 2,
        "games_per_generation": 1,
        "learning_rate": 0.001,
        "optimizer": Adam,
        "action_function": action_as_onehot,
        "value_loss_function": CategoricalCrossentropyLoss(),
        "reward_loss_function": CategoricalCrossentropyLoss(),
        "policy_loss_function": CategoricalCrossentropyLoss(),
        "support_range": 31,
    }

    config = MuZeroConfig(config_dict, game_config)

    print("Initializing Agent...")
    agent = MuZeroAgent(env, config)

    print("Running training smoke test...")
    # Manually populate buffer to ensure it can learn
    for _ in range(3):
        agent.play_game()

    print(f"Buffer size: {agent.replay_buffer.size}")

    agent.train()
    print("Training loop completed successfully!")


if __name__ == "__main__":
    verify_agent_refactor()
