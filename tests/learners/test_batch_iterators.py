import pytest
import torch
from unittest.mock import MagicMock
from agents.learner.batch_iterators import (
    SingleBatchIterator,
    RepeatSampleIterator,
    PPOEpochIterator,
)

pytestmark = pytest.mark.unit


def test_single_batch_iterator_device_transfer():
    """Verify that SingleBatchIterator yields tensors on the correct device."""
    device = torch.device("cpu")
    replay_buffer = MagicMock()
    batch = {
        "observations": torch.randn(2, 4),
        "actions": torch.tensor([0, 1]),
        "rewards": torch.randn(2),
        "not_a_tensor": "meta_data",
    }
    replay_buffer.sample.return_value = batch

    iterator = SingleBatchIterator(replay_buffer, device)
    yielded_batches = list(iterator)

    assert len(yielded_batches) == 1
    yielded_batch = yielded_batches[0]

    assert yielded_batch["observations"].device.type == device.type
    assert yielded_batch["actions"].device.type == device.type
    assert yielded_batch["rewards"].device.type == device.type
    assert yielded_batch["not_a_tensor"] == "meta_data"
    assert torch.is_tensor(yielded_batch["observations"])


def test_repeat_sample_iterator_device_transfer():
    """Verify that RepeatSampleIterator yields N batches on the correct device."""
    device = torch.device("cpu")
    num_iterations = 3
    replay_buffer = MagicMock()
    batch = {"data": torch.randn(2, 2)}
    replay_buffer.sample.return_value = batch

    iterator = RepeatSampleIterator(replay_buffer, num_iterations, device)
    yielded_batches = list(iterator)

    assert len(yielded_batches) == num_iterations
    for b in yielded_batches:
        assert b["data"].device.type == device.type


def test_ppo_epoch_iterator_lazy_gpu():
    """Verify that PPOEpochIterator yields tensors on the correct device."""
    device = torch.device("cpu")
    num_samples = 100
    num_epochs = 2
    num_minibatches = 10

    replay_buffer = MagicMock()
    full_batch = {
        "observations": torch.randn(num_samples, 4),
        "actions": torch.randint(0, 2, (num_samples,)),
        "rewards": torch.randn(num_samples),
    }
    replay_buffer.sample.return_value = full_batch

    iterator = PPOEpochIterator(
        replay_buffer=replay_buffer,
        num_epochs=num_epochs,
        num_minibatches=num_minibatches,
        device=device,
    )

    yielded_batches = list(iterator)
    assert len(yielded_batches) == num_epochs * num_minibatches

    for sub_batch in yielded_batches:
        assert sub_batch["observations"].device.type == device.type
        assert sub_batch["observations"].shape[0] == num_samples // num_minibatches
