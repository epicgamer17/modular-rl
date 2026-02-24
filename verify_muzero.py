import sys

sys.path.append(".")
import time
import torch
from agents.random import RandomAgent
from agents.trainers.muzero_trainer import MuZeroTrainer
from configs.agents.muzero import MuZeroConfig
from configs.games.tictactoe import TicTacToeConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from stats.stats import StatTracker

device = "cpu"

params = {
    "num_simulations": 25,
    "training_steps": 200,
    "transfer_interval": 1,
    "minibatch_size": 16,
    "replay_buffer_size": 2000,
    "num_workers": 1,
    "multi_process": False,
    "search_batch_size": 0,
    "action_selector": {
        "base": {"type": "categorical", "kwargs": {"exploration": True}}
    },
    "backbone": {
        "type": "resnet",
        "norm_type": "batch",
        "activation": "relu",
        "filters": [16],
        "kernel_sizes": [3],
        "strides": [1],
        "residual_layers": [(16, 3, 1)],
    },
    "world_model_cls": MuzeroWorldModel,
}

game_config = TicTacToeConfig()
env = game_config.make_env()
config = MuZeroConfig(config_dict=params, game_config=game_config)

trainer = MuZeroTrainer(
    config,
    env,
    torch.device("cpu"),
    name="muzero_verification",
    stats=StatTracker(name="muzero_veri"),
    test_agents=[RandomAgent()],
)

trainer.checkpoint_interval = 1000
trainer.test_interval = 200
trainer.test_trials = 20

trainer.train()
