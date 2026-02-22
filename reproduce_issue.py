import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

import time
import torch
import torch.nn.functional as F
from agents.random import RandomAgent
from agents.trainers.muzero_trainer import MuZeroTrainer
from configs.agents.muzero import MuZeroConfig
from configs.games.tictactoe import TicTacToeConfig
from agents.tictactoe_expert import TicTacToeBestAgent
from modules.world_models.muzero_world_model import MuzeroWorldModel
from stats.stats import StatTracker

# Ensure we use CPU for fairness/comparibility or GPU if available
device = "cpu"  # or "cuda" if available
print(f"Using device: {device}")

params = {
    "num_simulations": 25,
    "per_alpha": 0.0,
    "per_beta": 0.0,
    "per_beta_final": 0.0,
    "n_step": 10,
    "root_dirichlet_alpha": 0.25,
    "training_steps": 2000,  # Reduced for faster reproduction
    "test_interval": 500,
    "test_trials": 20,
    # Architecture Config (Shared settings)
    "architecture": {
        "residual_layers": [(24, 3, 1)],
    },
    # Main Backbone (Representation/Dynamics)
    "backbone": {
        "type": "resnet",
        "filters": [24],
        "kernel_sizes": [3],
        "strides": [1],
    },
    # Specialized Heads
    "reward_head": {
        "neck": {
            "type": "resnet",
            "filters": [16],
            "kernel_sizes": [1],
            "strides": [1],
        },
        "output_strategy": {"type": "regression"},
    },
    "value_head": {
        "neck": {
            "type": "resnet",
            "filters": [16],
            "kernel_sizes": [1],
            "strides": [1],
        },
        "output_strategy": {"type": "regression"},
    },
    "policy_head": {
        "neck": {
            "type": "resnet",
            "filters": [16],
            "kernel_sizes": [1],
            "strides": [1],
        }
    },
    "to_play_head": {
        "neck": {
            "type": "resnet",
            "filters": [16],
            "kernel_sizes": [1],
            "strides": [1],
        }
    },
    # Legacy fields that are still parsed for now
    "known_bounds": [-1, 1],
    "minibatch_size": 8,
    "replay_buffer_size": 100000,
    "gumbel": False,
    "gumbel_m": 16,
    "action_selector": {
        "base": {"type": "categorical"},
        "decorators": [{"type": "mcts"}],
    },
    "policy_loss_function": F.cross_entropy,
    "multi_process": False,
    "num_workers": 1,
    "world_model_cls": MuzeroWorldModel,
    "search_batch_size": 0,
    "use_virtual_mean": False,
    "virtual_loss": 3.0,
}

game_config = TicTacToeConfig()

print("--- Running MuZero Reproduction ---")
params_batched = params.copy()
params_batched["search_batch_size"] = 5
params_batched["use_virtual_mean"] = True
params_batched["model_name"] = "muzero_reproduction"

env_batch = TicTacToeConfig().make_env()
config_batch = MuZeroConfig(config_dict=params_batched, game_config=game_config)
config_batch.search_batch_size = 5  # Explicitly set

trainer = MuZeroTrainer(
    config_batch,
    env_batch,
    torch.device(device),
    stats=StatTracker(model_name="muzero_reproduction"),
    test_agents=[RandomAgent(), TicTacToeBestAgent()],
)
trainer.checkpoint_interval = 2000
trainer.test_interval = 500
trainer.test_trials = 20

start_time = time.time()
trainer.train()
end_time = time.time()
print(f"MuZero Reproduction Time: {end_time - start_time:.2f}s")
