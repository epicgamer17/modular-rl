import pytest
import numpy as np
from replay_buffers.writers import ReservoirWriter

# MANDATORY: Module-level pytest marker
pytestmark = pytest.mark.unit


def test_reservoir_writer_batch_unimplemented():
    writer = ReservoirWriter(max_size=10)

    # Verify that batch storing raises the explicit NotImplementedError
    with pytest.raises(NotImplementedError, match="Batch store not implemented"):
        writer.store_batch(5)


def test_reservoir_writer_capacity_and_calls():
    # Enforce determinism for the random reservoir logic
    np.random.seed(42)
    writer = ReservoirWriter(max_size=5)

    # Add 10 items to a buffer with a max_size of 5
    for _ in range(10):
        writer.store()

    # The size should cap at 5, but add_calls should track all 10 attempts
    assert writer.size == 5
    assert writer.add_calls == 10
