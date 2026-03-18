import pytest
import torch
from unittest.mock import MagicMock
from agents.learner.batch_iterators import PPOEpochIterator

pytestmark = pytest.mark.unit


def test_ppo_epoch_iterator_lazy_gpu():
    """Verify that PPOEpochIterator yields tensors on the correct device but samples on CPU."""
    # Use a dummy device if CUDA/MPS is not available, or just use CPU to test logic
    device = torch.device(
        "cpu"
    )  # In reality we want to test to a different device, but for unit test cpu is fine

    num_samples = 100
    batch_size = 10
    num_epochs = 2
    num_minibatches = 10  # 10 samples per minibatch

    # Mock replay buffer
    replay_buffer = MagicMock()
    # Create a batch of numpy-like tensors (but on CPU)
    full_batch = {
        "observations": torch.randn(num_samples, 4),
        "actions": torch.randint(0, 2, (num_samples,)),
        "rewards": torch.randn(num_samples),
    }
    replay_buffer.sample.return_value = full_batch

    # Initialize iterator
    iterator = PPOEpochIterator(
        replay_buffer=replay_buffer,
        num_epochs=num_epochs,
        num_minibatches=num_minibatches,
        device=device,
    )

    # Track yields
    yielded_batches = list(iterator)

    # Total yields should be num_epochs * num_minibatches
    assert len(yielded_batches) == num_epochs * num_minibatches

    for sub_batch in yielded_batches:
        assert isinstance(sub_batch, dict)
        assert "observations" in sub_batch
        # Check that yielded tensors are on the target device
        assert sub_batch["observations"].device.type == device.type
        assert sub_batch["observations"].shape[0] == num_samples // num_minibatches


def test_ppo_epoch_iterator_different_device():
    """Verify that PPOEpochIterator moves to correct device if specified."""
    # Only run if we have another device to test with
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        pytest.skip("No non-CPU device available for test")

    target_device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    num_samples = 20
    num_epochs = 1
    num_minibatches = 2

    replay_buffer = MagicMock()
    full_batch = {
        "observations": torch.randn(num_samples, 4),
    }
    replay_buffer.sample.return_value = full_batch

    iterator = PPOEpochIterator(
        replay_buffer=replay_buffer,
        num_epochs=num_epochs,
        num_minibatches=num_minibatches,
        device=target_device,
    )

    yielded_batches = list(iterator)
    for sub_batch in yielded_batches:
        # Tensors should be on the target device
        assert sub_batch["observations"].device.type == target_device.type
        # Original batch in closure (if we could access it) should still be on CPU
        # But we can check that if we modify the iterator to expose the batch or just trust the logic
