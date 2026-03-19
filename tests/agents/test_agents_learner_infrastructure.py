import pytest
import torch

from agents.learner.callbacks import LatentMetricsCallback
from utils.telemetry import finalize_metrics
from learners.losses.losses import SigmaLoss

pytestmark = pytest.mark.unit


def test_sigma_loss_emits_stochastic_metrics(make_cartpole_config):
    """
    Test that SigmaLoss emits stochastic telemetry directly into context metrics.
    """
    config = make_cartpole_config(stochastic=True)
    loss_module = SigmaLoss(config=config, device=torch.device("cpu"))

    predictions = {
        "chance_logits": torch.randn(8, 10),
    }
    targets = {
        "chance_codes": torch.zeros(8, 1).long(),
    }
    context = {"metrics": {}, "has_valid_action_mask": torch.ones(8, 1)}

    loss_module.compute_loss(
        predictions=predictions,
        targets=targets,
        context=context,
        k=0,
    )

    data = finalize_metrics(context["metrics"])
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
