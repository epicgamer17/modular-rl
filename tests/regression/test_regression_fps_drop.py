import time

import pytest
import torch

from agents.trainers.muzero_trainer import MuZeroTrainer
from configs.agents.muzero import MuZeroConfig
from configs.games.cartpole import CartPoleConfig
from modules.models.world_model import WorldModel
from stats.stats import StatTracker


@pytest.mark.regression
def test_regression_fps_drop():
    """
    Guardrail against severe training-loop throughput regressions.
    Ensures a short MuZero run completes and reports positive actor FPS.
    """
    params = {
        "executor_type": "local",
        "num_simulations": 2,
        "training_steps": 2,
        "transfer_interval": 50,
        "min_replay_buffer_size": 1,
        "minibatch_size": 4,
        "num_minibatches": 1,
        "replay_buffer_size": 128,
        "num_workers": 1,
        "known_bounds": [-1, 1],
        "action_selector": {
            "base": {"type": "categorical", "kwargs": {"exploration": True}}
        },
        "backbone": {"type": "dense", "widths": [16]},
        "reward_head": {
            "neck": {"type": "dense", "widths": [8]},
            "output_strategy": {"type": "scalar"},
        },
        "value_head": {
            "neck": {"type": "dense", "widths": [8]},
            "output_strategy": {"type": "scalar"},
        },
        "policy_head": {"neck": {"type": "dense", "widths": [8]}},
        "to_play_head": {"neck": {"type": "dense", "widths": [8]}},
        "world_model_cls": WorldModel,
    }

    game_config = CartPoleConfig()
    env = game_config.make_env()
    config = MuZeroConfig(config_dict=params, game_config=game_config)

    trainer = MuZeroTrainer(
        config,
        env,
        torch.device("cpu"),
        name="regression_fps_drop",
        stats=StatTracker(name="regression_fps_drop"),
    )
    trainer._save_checkpoint = lambda: None
    trainer.checkpoint_interval = config.training_steps + 1
    trainer.test_interval = config.training_steps + 1

    start_time = time.time()
    try:
        trainer.train()
    finally:
        if getattr(trainer, "executor", None) is not None:
            trainer.executor.stop()
        env.close()

    elapsed = time.time() - start_time
    stats = trainer.stats.get_data()
    actor_fps = [float(v) for v in stats.get("actor_fps", [])]

    assert trainer.training_step == config.training_steps
    assert elapsed < 60, f"Short training run took too long: {elapsed:.2f}s"
    assert actor_fps, "actor_fps should be recorded for throughput regression checks"
    assert min(actor_fps) > 0.0, f"actor_fps contains non-positive values: {actor_fps}"
