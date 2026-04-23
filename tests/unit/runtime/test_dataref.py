import pytest
import torch
from runtime.dataref import DataRef, StorageLocation, BufferRef, StreamRef

pytestmark = pytest.mark.unit

def test_dataref_creation_and_location():
    """Verify DataRef tracks its initial location."""
    data = torch.randn(10)
    ref = DataRef(data, location=StorageLocation.CPU)
    
    assert ref.location == StorageLocation.CPU
    assert len(ref.transfer_history) == 1
    assert ref.transfer_history[0]["reason"] == "creation"

def test_dataref_movement():
    """Verify DataRef can move between locations and tracks history."""
    data = torch.randn(10)
    ref = DataRef(data, location=StorageLocation.CPU)
    
    # Simulate move (actual GPU move requires CUDA, so we test the logic)
    ref.move_to(StorageLocation.SHARED_MEMORY)
    
    assert ref.location == StorageLocation.SHARED_MEMORY
    assert len(ref.transfer_history) == 2
    assert ref.transfer_history[1]["from"] == "cpu"
    assert ref.transfer_history[1]["to"] == "shared_memory"

def test_dataref_pytorch_integration():
    """Verify PyTorch tensor movement logic."""
    data = torch.randn(10)
    ref = DataRef(data, location=StorageLocation.CPU)
    
    # We can't easily test CUDA movement in CI without a GPU,
    # but we can verify it doesn't crash and handles the state correctly.
    if torch.cuda.is_available():
        ref.move_to(StorageLocation.GPU)
        assert ref.location == StorageLocation.GPU
        assert ref.data.is_cuda
        
        ref.move_to(StorageLocation.CPU)
        assert ref.location == StorageLocation.CPU
        assert not ref.data.is_cuda

def test_buffer_stream_inheritance():
    """Verify subclasses inherit location-aware behavior."""
    b_ref = BufferRef(torch.zeros(5), location=StorageLocation.CPU)
    s_ref = StreamRef(torch.ones(5), location=StorageLocation.CPU)
    
    assert b_ref.location == StorageLocation.CPU
    assert s_ref.location == StorageLocation.CPU
    
    b_ref.move_to(StorageLocation.REMOTE)
    assert b_ref.location == StorageLocation.REMOTE
    assert len(b_ref.transfer_history) == 2
