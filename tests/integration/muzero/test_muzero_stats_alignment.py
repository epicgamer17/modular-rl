import torch
import numpy as np
import pytest
from agents.trainers.muzero_trainer import MuZeroTrainer
from stats.stats import StatTracker
from configs.agents.muzero import MuZeroConfig

pytestmark = pytest.mark.integration

class MockAgent:
    """Mock agent to satisfy test_agents list."""
    def __init__(self, name):
        self.name = name

def test_muzero_trainer_stats_alignment(make_muzero_config_dict, tictactoe_game_config):
    """
    Standardized integration test to confirm that:
    1. Training scores (MP) are correctly mapped to 'score' with 'p0', 'p1', 'avg'.
    2. Evaluation scores (MP) are correctly mapped to 'test_score' with 'p0', 'p1', 'avg'.
    3. VS Agent scores are correctly mapped to 'vs_expert_score' with 'p0', 'p1', 'avg'.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    
    # 1. Setup real config for TicTacToe
    config_dict = make_muzero_config_dict(
        minibatch_size=2,
        unroll_steps=1,
        num_simulations=5,
        executor_type="local",
        # Ensure heads matches MuZero requirements
        to_play_head={"output_strategy": {"type": "categorical"}},
    )
    
    config = MuZeroConfig(config_dict, tictactoe_game_config)
    
    # 2. Get the real environment
    env = tictactoe_game_config.env_factory()
    
    # 3. Initialize Stats and Mock Agent
    stats = StatTracker(name="test_muzero")
    test_agent = MockAgent(name="expert")
    
    # 4. Initialize Trainer
    trainer = MuZeroTrainer(
        config=config,
        env=env,
        device=device,
        stats=stats,
        test_agents=[test_agent]
    )
    
    # 5. Verify initialized keys
    trainer._setup_stats()
    expected_score_subkeys = {"p0", "p1", "avg"}
    
    assert "score" in stats.stats, "Missing 'score' key in stats"
    assert set(stats.stats["score"].keys()) == expected_score_subkeys
    
    assert "test_score" in stats.stats, "Missing 'test_score' key in stats"
    assert set(stats.stats["test_score"].keys()) == expected_score_subkeys
    
    assert "vs_expert_score" in stats.stats, "Missing 'vs_expert_score' key in stats"
    assert set(stats.stats["vs_expert_score"].keys()) == expected_score_subkeys
    
    # 6. Simulate Training Results (Multi-player batch scores)
    mock_rollout_results = [
        {
            "batch_scores": [np.array([1.0, -1.0]), np.array([0.0, 1.0])],
            "batch_lengths": [10, 5],
            "fps": 100.0
        }
    ]
    trainer._record_collection_metrics(mock_rollout_results)
    
    # Verify values for first episode
    assert stats.stats["score"]["p0"][0] == 1.0
    assert stats.stats["score"]["p1"][0] == -1.0
    assert stats.stats["score"]["avg"][0] == 0.0
    
    # Verify values for second episode
    assert stats.stats["score"]["p0"][1] == 0.0
    assert stats.stats["score"]["p1"][1] == 1.0
    assert stats.stats["score"]["avg"][1] == 0.5
    
    # 7. Simulate Evaluation Results
    mock_eval_res = {
        "score": {
            "avg": 0.5,
            "p0": 1.0,
            "p1": 0.0
        },
        "vs_expert_score": {
            "avg": -0.2,
            "p0": 0.0,
            "p1": -0.4
        },
        "avg_length": 8.0,
        "episodes_completed": 10
    }
    
    trainer._process_test_results(mock_eval_res, step=100)
    
    assert stats.stats["test_score"]["avg"][-1] == 0.5
    assert stats.stats["test_score"]["p0"][-1] == 1.0
    assert stats.stats["vs_expert_score"]["p1"][-1] == -0.4
    
    print("\n[SUCCESS] MuZero Trainer stats are correctly aligned with TicTacToe environment.")

if __name__ == "__main__":
    from tests.conftest import base_muzero_config_dict, make_muzero_config_dict, tictactoe_game_config
    
    # Manual fixture orchestration
    base_dict = base_muzero_config_dict()
    make_fn = make_muzero_config_dict(base_dict)
    game_cfg = tictactoe_game_config()
    
    test_muzero_trainer_stats_alignment(make_fn, game_cfg)
