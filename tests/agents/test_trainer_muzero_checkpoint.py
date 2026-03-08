import pytest
import torch
import numpy as np
import os
from agents.trainers.muzero_trainer import MuZeroTrainer
from configs.agents.muzero import MuZeroConfig

# MANDATORY: Module-level pytest marker
pytestmark = pytest.mark.slow


def test_muzero_trainer_checkpointing(
    make_muzero_config_dict, cartpole_game_config, tmp_path
):
    # Enforce strict determinism
    torch.manual_seed(42)
    np.random.seed(42)

    # Inject real configuration, forcing synchronous execution for safety
    config_dict = make_muzero_config_dict(num_workers=0, multi_process=False)
    config = MuZeroConfig(config_dict, cartpole_game_config)

    # Initialize a lightweight dummy environment
    env = cartpole_game_config.make_env()

    # Initialize the trainer (this hits dozens of lines in both trainer files)
    trainer = MuZeroTrainer(config=config, env=env, device=torch.device("cpu"))

    # Test the Happy Path: Saving a checkpoint
    checkpoint_file = tmp_path / "test_muzero_checkpoint.pt"
    trainer.save_checkpoint(str(checkpoint_file))
    assert os.path.exists(checkpoint_file)

    # Test the Unhappy Path: Attempt to load from a file that doesn't exist
    bad_file = tmp_path / "does_not_exist.pt"
    with pytest.raises(FileNotFoundError):
        trainer.load_checkpoint(str(bad_file))
