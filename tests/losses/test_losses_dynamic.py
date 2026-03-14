import pytest
import torch
from unittest.mock import MagicMock
from losses.losses import LossModule, LossPipeline
from modules.world_models.inference_output import LearningOutput

pytestmark = pytest.mark.unit


class MockLoss(LossModule):
    def __init__(self, config, device, name, req_preds, req_targets):
        super().__init__(config, device)
        self.name = name  # Override the default class name
        self._req_preds = req_preds
        self._req_targets = req_targets
        self.compute_call_count = 0

    @property
    def required_predictions(self):
        return self._req_preds

    @property
    def required_targets(self):
        return self._req_targets

    def compute_loss(self, predictions, targets, context, k=0):
        self.compute_call_count += 1
        # Return something of shape (B,)
        batch_size = next(iter(predictions.values())).shape[0]
        return torch.ones(batch_size, device=self.device)


def test_loss_pipeline_validation():
    config = MagicMock()
    config.minibatch_size = 2
    device = torch.device("cpu")

    module1 = MockLoss(config, device, "Loss1", {"p1"}, {"t1"})
    module2 = MockLoss(config, device, "Loss2", {"p2"}, {"t2"})
    pipeline = LossPipeline([module1, module2])

    # Validation Success
    pipeline.validate_dependencies({"p1", "p2", "extra"}, {"t1", "t2"})

    # Validation Failure - Missing Prediction
    with pytest.raises(ValueError) as excinfo:
        pipeline.validate_dependencies({"p1"}, {"t1", "t2"})
    assert "Module Loss2 missing required predictions: {'p2'}" in str(excinfo.value)

    # Validation Failure - Missing Target
    with pytest.raises(ValueError) as excinfo:
        pipeline.validate_dependencies({"p1", "p2"}, {"t1"})
    assert "Module Loss2 missing required targets: {'t2'}" in str(excinfo.value)


def test_loss_pipeline_execution():
    torch.manual_seed(42)
    device = torch.device("cpu")
    config = MagicMock()
    config.minibatch_size = 2
    config.mask_absorbing = False
    config.support_range = None

    # Setup modules - use valid LearningOutput fields
    module1 = MockLoss(config, device, "Loss1", {"values"}, {"returns"})
    # Mocking compute_loss to return a constant for easy testing
    module1.compute_loss = MagicMock(
        return_value=torch.tensor([1.0, 1.0], device=device)
    )

    pipeline = LossPipeline([module1])

    # Setup data using valid LearningOutput fields
    predictions = LearningOutput(
        values=torch.tensor([[0.1, 0.2], [0.1, 0.2]], device=device)
    )
    targets = LearningOutput(
        values=torch.tensor([[0.3, 0.4], [0.3, 0.4]], device=device)
    )
    weights = torch.tensor([1.0, 1.0], device=device)
    gradient_scales = torch.tensor([[1.0, 0.5]], device=device)  # K=1 (2 steps)
    context = {}

    loss_mean, loss_dict, priorities = pipeline.run(
        predictions=predictions,
        targets=targets._asdict(),
        context=context,
        weights=weights,
        gradient_scales=gradient_scales,
    )

    # 2 steps (k=0, k=1)
    # k=0: loss_k = [1, 1], scale=1.0, weighted=[1, 1], sum=2
    # k=1: loss_k = [1, 1], scale=0.5, weighted=[0.5, 0.5], sum=1
    # total_sum = 4.0
    # loss_mean = 4.0 / 2 = 2.0

    assert torch.allclose(loss_mean, torch.tensor(2.0, device=device))
    assert loss_dict["Loss1"] == 2.0  # (1+1 + 1+1) / 2 = 2.0
    assert module1.compute_loss.call_count == 2


def test_priority_calculation():
    device = torch.device("cpu")
    config = MagicMock()
    config.minibatch_size = 2
    config.support_range = None

    mock_module = MagicMock(config=config, device=device)
    mock_module.compute_priority.return_value = None
    pipeline = LossPipeline([mock_module])

    # Case 1: Scalar values
    preds_k = {"q_values": torch.tensor([1.0, 3.0], device=device)}
    targets_k = {"q_values": torch.tensor([2.0, 2.0], device=device)}

    # Mock compute_priority for Case 1
    mock_module.compute_priority.side_effect = lambda preds, targets, ctx, k: torch.abs(targets["q_values"] - preds["q_values"])
    priorities = pipeline._calculate_priorities(preds_k, targets_k, {}, config, device)
    # abs(2-1) = 1, abs(2-3) = 1
    assert torch.allclose(priorities, torch.tensor([1.0, 1.0], device=device))

    # Case 2: Support values (MuZero style)
    config.support_range = (-10, 10)

    preds_k = {
        "values": torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device)
    }  # Mock distributions
    targets_k = {"values": torch.tensor([0.0, 10.0], device=device)}

    # We need to make sure the mock support used by pipeline matches
    import modules.utils

    original_sts = modules.utils.support_to_scalar
    # Mocking side effects for the call in _calculate_priorities
    modules.utils.support_to_scalar = MagicMock(
        side_effect=[torch.tensor([0.0, 10.0], device=device)]
    )

    try:
        # Mock compute_priority for Case 2
        mock_module.compute_priority.side_effect = lambda preds, targets, ctx, k: torch.abs(targets["values"] - modules.utils.support_to_scalar(preds["values"], config.support_range))
        priorities = pipeline._calculate_priorities(preds_k, targets_k, {}, config, device)
        # abs(0-0) = 0, abs(10-10) = 0
        assert torch.allclose(priorities, torch.tensor([0.0, 0.0], device=device))
    finally:
        modules.utils.support_to_scalar = original_sts


def test_step_extraction():
    pipeline = LossPipeline([MagicMock()])

    # K=1 (expected_steps=2)
    # Use valid LearningOutput fields - values has shape (B, K+1)
    predictions = LearningOutput(
        values=torch.tensor([[1, 2], [3, 4]]),  # (B, K+1)
    )

    # k=0
    data_0 = pipeline._extract_step_data(predictions._asdict(), 0, 2)
    assert torch.equal(data_0["values"], torch.tensor([1, 3]))

    # k=1
    data_1 = pipeline._extract_step_data(predictions._asdict(), 1, 2)
    assert torch.equal(data_1["values"], torch.tensor([2, 4]))


def test_masking_execution():
    device = torch.device("cpu")
    config = MagicMock()
    config.minibatch_size = 2
    config.mask_absorbing = True
    config.support_range = None

    # Use valid LearningOutput fields
    module1 = MockLoss(config, device, "Loss1", {"values"}, {"returns"})
    module1.get_mask = MagicMock(return_value=torch.tensor([1.0, 0.0], device=device))
    module1.compute_loss = MagicMock(
        return_value=torch.tensor([10.0, 10.0], device=device)
    )

    pipeline = LossPipeline([module1])

    predictions = LearningOutput(values=torch.tensor([[0.1], [0.1]], device=device))
    targets = LearningOutput(values=torch.tensor([[0.3], [0.3]], device=device))
    weights = torch.tensor([1.0, 1.0], device=device)
    gradient_scales = torch.tensor([[1.0]], device=device)
    context = {}

    loss_mean, loss_dict, priorities = pipeline.run(
        predictions=predictions,
        targets=targets._asdict(),
        context=context,
        weights=weights,
        gradient_scales=gradient_scales,
    )

    # loss_k = [10, 10], mask = [1, 0] -> masked = [10, 0]
    # sum = 10, mean = 5.0
    assert torch.allclose(loss_mean, torch.tensor(5.0, device=device))
