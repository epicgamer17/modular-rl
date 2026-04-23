import pytest
import torch
from runtime.dataref import DataRef, BufferRef, StreamRef

pytestmark = pytest.mark.unit

def test_dataref_no_copy():
    """Verify that DataRef does not copy the underlying tensor."""
    original = torch.randn(10, 10)
    ref = DataRef(original)
    
    assert ref.data is original, "DataRef should maintain object identity with the original tensor."
    assert id(ref.data) == id(original)

def test_dataref_shape_preservation():
    """Verify that DataRef preserves tensor shape metadata."""
    shape = (2, 3, 4)
    original = torch.zeros(shape)
    ref = DataRef(original)
    
    assert ref.data.shape == shape
    assert ref.data.dtype == original.dtype

def test_dataref_types():
    """Verify different DataRef subtypes."""
    t = torch.ones(5)
    
    buf_ref = BufferRef(t)
    stream_ref = StreamRef(t)
    
    assert isinstance(buf_ref, DataRef)
    assert isinstance(stream_ref, DataRef)
    assert buf_ref.data is t
    assert stream_ref.data is t

if __name__ == "__main__":
    test_dataref_no_copy()
    test_dataref_shape_preservation()
    test_dataref_types()
    print("Test 3.2 Passed!")
