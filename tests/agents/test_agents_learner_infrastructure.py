import pytest
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

from configs.agents.ppo import PPOConfig
from agents.learners.callbacks import StochasticMetricsCallback, LatentMetricsCallback
from modules.agent_nets.modular import ModularAgentNetwork
from stats.stats import StatTracker

pytestmark = pytest.mark.unit


def test_stochastic_metrics_callback_smoke(make_cartpole_config):
    """
    Test the StochasticMetricsCallback by manually firing it with dummy data.
    """
    config = make_cartpole_config(stochastic=True)
    callback = StochasticMetricsCallback()

    class MockLearner:
        def __init__(self):
            self.config = config
            self.device = torch.device("cpu")
            self.training_step = 0

    learner = MockLearner()
    stats = StatTracker(name="test")

    predictions = {
        "chance_codes": torch.randn(8, 2, 1, 10),
    }
    targets = {
        "chance_codes": torch.zeros(8, 2, 1).long(),
    }

    callback.on_step_end(
        learner=learner,
        predictions=predictions,
        targets=targets,
        loss_dict={"loss": 0.5},
        stats=stats,
    )

    data = stats.get_data()
    assert "num_codes" in data
    assert "chance_probs" in data
    assert "chance_entropy" in data


def test_latent_metrics_callback_smoke(make_cartpole_config):
    """
    Test the LatentMetricsCallback by manually firing it with dummy data.
    """
    config = make_cartpole_config(latent_viz_interval=1, latent_viz_method="pca")
    callback = LatentMetricsCallback(
        viz_interval=config.latent_viz_interval, viz_method="pca"
    )

    class MockLearner:
        def __init__(self):
            self.config = config
            self.device = torch.device("cpu")
            self.training_step = 0

    learner = MockLearner()
    stats = StatTracker(name="test")

    predictions = {
        "latent_states": torch.randn(8, 2, 128),
    }
    targets = {
        "actions": torch.zeros(8, 2, 1).long(),
    }

    callback.on_step_end(
        learner=learner,
        predictions=predictions,
        targets=targets,
        loss_dict={"loss": 0.5},
        stats=stats,
    )

    # Smoke test: ensuring it doesn't crash is often enough for visualization callbacks
    # unless we want to inspect the StatTracker interna
