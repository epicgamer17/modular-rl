import pytest
import torch
import torch.nn as nn
from agents.trainers.muzero_trainer import MuZeroTrainer
from configs.agents.muzero import MuZeroConfig
from modules.embeddings.actions.spatial import SpatialActionEmbedding

pytestmark = pytest.mark.integration

def test_tictactoe_uses_spatial_embedding(make_muzero_config_dict, tictactoe_game_config):
    """
    Verify that for TicTacToe (which is an image-based game),
    MuZero correctly selects the SpatialActionEmbedding.
    """
    # 1. Setup MuZero config for TicTacToe with spatial backbones
    # Using Conv backbones ensures the latent state remains spatial (C, H, W)
    config_dict = make_muzero_config_dict(
        representation_backbone={
            "type": "conv",
            "filters": [16],
            "kernel_sizes": [3],
            "strides": [1],
            "norm_type": "none"
        },
        dynamics_backbone={
            "type": "conv",
            "filters": [16],
            "kernel_sizes": [3],
            "strides": [1],
            "norm_type": "none"
        },
        # Minimal training setup
        minibatch_size=2,
        unroll_steps=1,
        executor_type="local",
        to_play_head={"output_strategy": {"type": "categorical"}},
    )
    # Ensure is_image is True in the game config (it should be for TicTacToe)
    assert tictactoe_game_config.is_image is True
    
    config = MuZeroConfig(config_dict, tictactoe_game_config)
    
    # 2. Get the environment
    env = tictactoe_game_config.env_factory()
    
    # 3. Initialize Trainer
    device = torch.device("cpu")
    trainer = MuZeroTrainer(config, env, device)
    
    # 4. Inspect the embedding module
    # Path: trainer.agent_network.components["world_model"].dynamics_pipeline.dynamics_fusion.encoder.embedding_module
    wm = trainer.agent_network.components["world_model"]
    
    # Check if dynamics_pipeline exists and has dynamics_fusion
    assert hasattr(wm, "dynamics_pipeline"), "WorldModel missing dynamics_pipeline"
    dp = wm.dynamics_pipeline
    
    assert hasattr(dp, "dynamics_fusion"), "DynamicsPipeline missing dynamics_fusion"
    fusion = dp.dynamics_fusion
    
    assert hasattr(fusion, "encoder"), "ActionFusion missing encoder"
    encoder = fusion.encoder
    
    assert hasattr(encoder, "embedding_module"), "ActionEncoder missing embedding_module"
    embedding_module = encoder.embedding_module
    
    # Verify it is SpatialActionEmbedding
    assert isinstance(embedding_module, SpatialActionEmbedding), \
        f"Expected SpatialActionEmbedding for TicTacToe, got {type(embedding_module)}"
    
    # Verify dimensions (TicTacToe is 3x3)
    assert embedding_module.h == 3, f"Expected height 3, got {embedding_module.h}"
    assert embedding_module.w == 3, f"Expected width 3, got {embedding_module.w}"
    assert embedding_module.num_actions == 9, f"Expected 9 actions, got {embedding_module.num_actions}"

    print("\n[SUCCESS] TicTacToe is correctly using SpatialActionEmbedding.")


def test_cartpole_uses_efficientzero_embedding(make_muzero_config_dict, cartpole_game_config):
    """
    Verify that for CartPole (which is a vector-based game),
    MuZero correctly selects the EfficientZeroActionEmbedding.
    """
    from modules.embeddings.actions.efficient_zero import EfficientZeroActionEmbedding

    # 1. Setup MuZero config for CartPole
    config_dict = make_muzero_config_dict(
        representation_backbone={"type": "mlp", "layers": [16]},
        dynamics_backbone={"type": "mlp", "layers": [16]},
        executor_type="local",
    )
    # Ensure is_image is False in the game config
    assert cartpole_game_config.is_image is False
    
    config = MuZeroConfig(config_dict, cartpole_game_config)
    
    # 2. Get the environment
    env = cartpole_game_config.env_factory()
    
    # 3. Initialize Trainer
    device = torch.device("cpu")
    trainer = MuZeroTrainer(config, env, device)
    
    # 4. Inspect the embedding module
    wm = trainer.agent_network.components["world_model"]
    embedding_module = wm.dynamics_pipeline.dynamics_fusion.encoder.embedding_module
    
    print(f"Detected embedding module: {type(embedding_module)}")
    
    assert isinstance(embedding_module, EfficientZeroActionEmbedding), \
        f"Expected EfficientZeroActionEmbedding for CartPole, got {type(embedding_module)}"
    
    assert embedding_module.num_actions == 2, f"Expected 2 actions, got {embedding_module.num_actions}"

    print("\n[SUCCESS] CartPole is correctly using EfficientZeroActionEmbedding.")


def test_continuous_uses_continuous_embedding(make_muzero_config_dict):
    """
    Verify that for a continuous action space,
    MuZero correctly selects the ContinuousActionEmbedding.
    """
    from modules.embeddings.actions.continuous import ContinuousActionEmbedding
    from configs.games.game import GameConfig
    from unittest.mock import MagicMock

    # 1. Setup synthetic continuous game config
    import numpy as np
    import gymnasium as gym
    class MockEnv:
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
            self.spec = None
            self.metadata = {}
        def reset(self, **kwargs):
            return np.zeros(8, dtype=np.float32), {}
        def close(self):
            pass

    class MockGameConfig(GameConfig):
        def __init__(self):
            self.is_discrete = False
            self.num_actions = 4
            self.is_image = False
            self.num_players = 1
            self.min_score = 0
            self.max_score = 1
            self.env_factory = lambda: MockEnv()
    
    continuous_game_config = MockGameConfig()
    mock_env = MockEnv()

    # 2. Setup MuZero config
    config_dict = make_muzero_config_dict(
        representation_backbone={"type": "mlp", "layers": [16]},
        dynamics_backbone={"type": "mlp", "layers": [16]},
        executor_type="local",
    )
    
    config = MuZeroConfig(config_dict, continuous_game_config)
    
    # 3. Initialize Trainer
    device = torch.device("cpu")
    trainer = MuZeroTrainer(config, mock_env, device)
    
    # 4. Inspect the embedding module
    wm = trainer.agent_network.components["world_model"]
    embedding_module = wm.dynamics_pipeline.dynamics_fusion.encoder.embedding_module
    
    print(f"Detected embedding module: {type(embedding_module)}")
    
    assert isinstance(embedding_module, ContinuousActionEmbedding), \
        f"Expected ContinuousActionEmbedding for continuous env, got {type(embedding_module)}"
    
    assert embedding_module.num_actions == 4, f"Expected 4 actions, got {embedding_module.num_actions}"

    print("\n[SUCCESS] Continuous env is correctly using ContinuousActionEmbedding.")
