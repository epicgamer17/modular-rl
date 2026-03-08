import pytest
from replay_buffers.writers import PPOWriter

pytestmark = pytest.mark.unit


def test_ppo_writer_single_overflow():
    # Initialize with a tiny max_size
    writer = PPOWriter(max_size=5)

    # Fill the buffer to the brim
    for _ in range(5):
        writer.store()

    # The 6th store should trigger the explicit IndexError
    with pytest.raises(IndexError, match="PPO Buffer Overflow"):
        writer.store()


def test_ppo_writer_batch_overflow():
    writer = PPOWriter(max_size=10)

    # Partially fill
    writer.store_batch(8)

    # Attempt to write a batch that exceeds remaining capacity
    with pytest.raises(IndexError, match="PPO Buffer Overflow"):
        writer.store_batch(5)
