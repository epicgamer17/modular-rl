import pytest
from replay_buffers.writers import SharedCircularWriter

pytestmark = pytest.mark.unit


def test_shared_circular_writer_state_and_clear():
    writer = SharedCircularWriter(max_size=10)

    assert writer.pointer == 0
    assert writer.size == 0

    # Store a few items
    idx1 = writer.store()
    idx2 = writer.store()

    # Verify indices and tensor-backed properties update correctly
    assert idx1 == 0
    assert idx2 == 1
    assert writer.pointer == 2
    assert writer.size == 2

    # Verify clearing correctly resets the underlying shared tensors
    writer.clear()
    assert writer.pointer == 0
    assert writer.size == 0
