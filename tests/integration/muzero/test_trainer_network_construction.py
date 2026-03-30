import pytest
import torch

from agents.trainers.muzero_trainer import MuZeroTrainer
from stats.stats import StatTracker
from configs.agents.muzero import MuZeroConfig

pytestmark = pytest.mark.integration

def test_muzero_trainer_builds_prediction_backbone(
    make_muzero_config_dict, tictactoe_game_config
):
    """
    Ensures that MuZeroTrainer parses `prediction_backbone` from the config
    and properly hands it to the manually-assembled AgentNetwork during its
    initialization process.
    """
    # 1. Provide minimal valid config dict with explicit prediction_backbone
    cfg_dict = make_muzero_config_dict(
        prediction_backbone={
            "type": "mlp",
            "widths": [16, 16],
            "norm_type": "none",
            "activation": "relu"
        },
        num_workers=0,  # Prevent actual Ray worker spawning
        multi_process=False,
        search_backend="python"
    )

    config = MuZeroConfig(cfg_dict, tictactoe_game_config)
    env = tictactoe_game_config.env_factory()

    # 2. Build Trainer
    trainer = MuZeroTrainer(
        config=config,
        env=env,
        device=torch.device("cpu"),
        name="test_trainer_prediction",
        stats=StatTracker("test_trainer_prediction"),
    )

    # 3. Assert network components
    assert hasattr(trainer, "agent_network"), "Trainer missing agent_network."
    assert "prediction" in trainer.agent_network.components, (
        "MuZeroTrainer failed to pass prediction_fn to AgentNetwork construction. "
        "This likely means prediction_backbone is being ignored."
    )
    
    pred_layer = trainer.agent_network.components["prediction"]
    assert hasattr(pred_layer, "output_shape"), "Prediction layer must expose output_shape."
    assert pred_layer.output_shape == (16,), f"Expected output_shape (16,), got {pred_layer.output_shape}."
