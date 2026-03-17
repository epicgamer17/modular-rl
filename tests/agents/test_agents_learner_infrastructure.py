import pytest
import torch

from agents.learners.callbacks import (
    LatentMetricsCallback,
    StochasticMetricsCallback,
    finalize_metrics,
)

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
    predictions = {
        "chance_codes": torch.randn(8, 2, 1, 10),
    }
    targets = {
        "chance_codes": torch.zeros(8, 2, 1).long(),
    }
    meta = {"metrics": {}}

    callback.on_step_end(
        learner=learner,
        predictions=predictions,
        targets=targets,
        loss_dict={"loss": 0.5},
        meta=meta,
    )

    data = finalize_metrics(meta["metrics"])
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
    predictions = {
        "latent_states": torch.randn(8, 2, 128),
    }
    targets = {
        "actions": torch.zeros(8, 2, 1).long(),
    }
    meta = {"metrics": {}}

    callback.on_step_end(
        learner=learner,
        predictions=predictions,
        targets=targets,
        loss_dict={"loss": 0.5},
        meta=meta,
    )

    metrics = finalize_metrics(meta["metrics"])
    assert "_latent_visualizations" in metrics
