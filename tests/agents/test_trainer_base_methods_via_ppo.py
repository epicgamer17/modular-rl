import pytest
import torch
import numpy as np
import os
from agents.trainers.ppo_trainer import PPOTrainer
from configs.agents.ppo import PPOConfig

pytestmark = pytest.mark.slow


def test_base_trainer_checkpointing(
    make_ppo_config_dict, cartpole_game_config, tmp_path
):
    torch.manual_seed(42)
    np.random.seed(42)

    config_dict = make_ppo_config_dict()
    config = PPOConfig(config_dict, cartpole_game_config)

    # Initialize child trainer to access base methods
    trainer = PPOTrainer(config=config)

    # Test the base save/load checkpoint logic using pytest's built-in tmp_path fixture
    checkpoint_file = tmp_path / "test_checkpoint.pt"

    # Save
    trainer.save_checkpoint(str(checkpoint_file))
    assert os.path.exists(checkpoint_file)

    # Load (Unhappy Path): Attempt to load from a non-existent file
    bad_file = tmp_path / "does_not_exist.pt"
    with pytest.raises(FileNotFoundError):
        trainer.load_checkpoint(str(bad_file))
