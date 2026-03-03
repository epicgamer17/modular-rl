import time
import torch

from agents.random import RandomAgent
from agents.trainers.muzero_trainer import MuZeroTrainer
from configs.agents.muzero import MuZeroConfig
from configs.games.tictactoe import TicTacToeConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from stats.stats import StatTracker

params = {
    "num_simulations": 10,
    "training_steps": 200,
    "transfer_interval": 1,
    "minibatch_size": 8,
    "replay_buffer_size": 1000,
    "num_workers": 2,
    "multi_process": True,
    "search_batch_size": 0,
    "known_bounds": [-1, 1],
    "action_selector": {
        "base": {"type": "categorical", "kwargs": {"exploration": True}}
    },
    "backbone": {
        "type": "dense",
        "widths": [32],
    },
    "reward_head": {
        "neck": {"type": "dense", "widths": [16]},
        "output_strategy": {"type": "scalar"},
    },
    "value_head": {
        "neck": {"type": "dense", "widths": [16]},
        "output_strategy": {"type": "scalar"},
    },
    "policy_head": {
        "neck": {"type": "dense", "widths": [16]},
    },
    "to_play_head": {
        "neck": {"type": "dense", "widths": [16]},
    },
    "world_model_cls": MuzeroWorldModel,
}

if __name__ == "__main__":
    game_config = TicTacToeConfig()
    env = game_config.make_env()
    config = MuZeroConfig(config_dict=params, game_config=game_config)
    trainer = MuZeroTrainer(
        config,
        env,
        torch.device("cpu"),
        name="muzero_fps_test",
        stats=StatTracker(name="muzero_fps_test"),
        test_agents=[RandomAgent()],
    )
    trainer.checkpoint_interval = 200
    trainer.test_interval = 50
    trainer.test_trials = 5
    trainer.train()
