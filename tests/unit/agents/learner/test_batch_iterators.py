import pytest
import torch
import numpy as np
from agents.learner.batch_iterators import PPOEpochIterator
from replay_buffers.modular_buffer import ModularReplayBuffer, BufferConfig

pytestmark = pytest.mark.unit

def test_ppo_epoch_iterator_shuffling_and_minibatch():
    """
    Tier 1: PPO Mini-batching Test.
    Verifies that the PPOEpochIterator correctly shuffles data and splits it
    into the requested number of mini-batches across multiple epochs.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Setup a simple PPO-style buffer
    max_size = 10
    config = [BufferConfig("observations", shape=(1,), dtype=torch.float32)]
    buffer = ModularReplayBuffer(max_size=max_size, buffer_configs=config, batch_size=max_size)
    
    # 2. Fill the buffer with unique values [0, 1, ..., 9]
    for i in range(max_size):
        buffer.store(observations=torch.tensor([float(i)]))
    
    assert buffer.size == max_size
    
    # 3. Initialize Iterator: 2 epochs, 2 mini-batches (size 5 each)
    num_epochs = 2
    num_minibatches = 2
    iterator = PPOEpochIterator(
        replay_buffer=buffer,
        num_epochs=num_epochs,
        num_minibatches=num_minibatches,
        device=torch.device("cpu")
    )
    
    # 4. Collect all yielded mini-batches
    yielded_batches = list(iterator)
    
    # 5. Verify total number of batches: epochs * minibatches = 4
    assert len(yielded_batches) == num_epochs * num_minibatches, f"Expected 4 batches, got {len(yielded_batches)}"
    
    # 6. Verify data coverage per epoch
    for epoch in range(num_epochs):
        epoch_batches = yielded_batches[epoch * num_minibatches : (epoch + 1) * num_minibatches]
        
        # Check sizes
        assert epoch_batches[0]["observations"].shape[0] == 5
        assert epoch_batches[1]["observations"].shape[0] == 5
        
        # Check coverage: all values [0..9] must appear exactly once
        epoch_values = torch.cat([b["observations"].flatten() for b in epoch_batches])
        sorted_values, _ = torch.sort(epoch_values)
        expected_values = torch.arange(max_size, dtype=torch.float32)
        
        torch.testing.assert_close(sorted_values, expected_values, msg=f"Epoch {epoch} did not cover all data points exactly once.")

    # 7. Verify shuffling: epoch 1 order should be different from epoch 2
    epoch0_order = torch.cat([b["observations"].flatten() for b in yielded_batches[0:2]])
    epoch1_order = torch.cat([b["observations"].flatten() for b in yielded_batches[2:4]])
    
    # With seed 42, they should be different. If they are the same, it's a shuffling bug.
    assert not torch.equal(epoch0_order, epoch1_order), "Data order was identical across epochs; shuffling might be broken."

def test_ppo_epoch_iterator_uneven_split():
    """
    Verifies that the iterator handles cases where num_samples is not divisible by num_minibatches.
    """
    torch.manual_seed(42)
    max_size = 11
    config = [BufferConfig("observations", shape=(1,), dtype=torch.float32)]
    buffer = ModularReplayBuffer(max_size=max_size, buffer_configs=config, batch_size=max_size)
    
    for i in range(max_size):
        buffer.store(observations=torch.tensor([float(i)]))
        
    # 11 samples, 3 mini-batches -> ceiling(11/3) = 4 per minibatch (mostly)
    # Expected sizes: [4, 4, 3]
    num_epochs = 1
    num_minibatches = 3
    iterator = PPOEpochIterator(
        replay_buffer=buffer,
        num_epochs=num_epochs,
        num_minibatches=num_minibatches,
        device=torch.device("cpu")
    )
    
    batches = list(iterator)
    assert len(batches) == 3
    sizes = [b["observations"].shape[0] for b in batches]
    assert sizes == [4, 4, 3], f"Expected sizes [4, 4, 3], got {sizes}"
    
    # Check coverage
    all_values = torch.cat([b["observations"].flatten() for b in batches])
    sorted_values, _ = torch.sort(all_values)
    torch.testing.assert_close(sorted_values, torch.arange(max_size, dtype=torch.float32))

def test_ppo_epoch_iterator_device_moving():
    """
    Verfies that the iterator correctly moves mini-batches to the target device.
    (Testing with CPU as default since CI might not have GPU/MPS, but checks the .device attribute).
    """
    device = torch.device("cpu")
    max_size = 4
    config = [BufferConfig("observations", shape=(1,), dtype=torch.float32)]
    buffer = ModularReplayBuffer(max_size=max_size, buffer_configs=config, batch_size=max_size)
    
    for i in range(max_size):
        buffer.store(observations=torch.tensor([float(i)]))
        
    iterator = PPOEpochIterator(
        replay_buffer=buffer,
        num_epochs=1,
        num_minibatches=2,
        device=device
    )
    
    for batch in iterator:
        assert batch["observations"].device.type == "cpu"
        # We also check that non-tensors are preserved (if any)
        # Replay buffer sample for PPO might include 'indices' which are numpy in this framework
        # but the iterator moves them only if they are tensors.
