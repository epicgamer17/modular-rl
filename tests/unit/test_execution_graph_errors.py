import torch
import pytest
from core.component import PipelineComponent
from core.contracts import Key, Observation, ShapeContract, WriteMode
from core.blackboard_engine import BlackboardEngine

pytestmark = pytest.mark.unit

class MockComponent(PipelineComponent):
    def __init__(self, name: str, requires=None, provides=None):
        self._name = name
        self._requires = requires or set()
        self._provides = provides or {}

    @property
    def requires(self): return self._requires
    @property
    def provides(self): return self._provides
    def execute(self, blackboard): return {}

def test_better_error_message_time_dim_mismatch():
    """Verifies the new structured error message for time_dim mismatch."""
    # Component A provides (B, T, 128)
    k_p = Key("data.x", Observation, shape=ShapeContract(
        symbolic=("B", "T", "C"),
        time_dim=1,
        event_shape=(128,),
        ndim=3
    ))
    
    # Component B requires (B, 128)
    k_c = Key("data.x", Observation, shape=ShapeContract(
        symbolic=("B", "C"),
        time_dim=None,
        event_shape=(128,),
        ndim=2
    ))
    
    c1 = MockComponent("Producer", provides={k_p: WriteMode.NEW})
    c2 = MockComponent("Consumer", requires={k_c})
    
    with pytest.raises(RuntimeError) as excinfo:
        BlackboardEngine(components=[c1, c2], device=torch.device("cpu"))
    
    error_msg = str(excinfo.value)
    
    # Check for structured parts
    assert "Component A (Key: data.x, Provider: [0] MockComponent)" in error_msg
    assert "provides: shape (B, T, C)" in error_msg
    assert "Component B (Key: data.x, Consumer: [1] MockComponent)" in error_msg
    assert "requires: shape (B, C)" in error_msg
    assert "Error:" in error_msg
    assert "Time dimension mismatch" in error_msg
    assert "sequence dimension (T)" in error_msg

def test_better_error_message_ndim_mismatch():
    """Verifies the new structured error message for ndim mismatch."""
    # Component A provides (B, 128)
    k_p = Key("data.x", Observation, shape=ShapeContract(ndim=2))
    
    # Component B requires (B, T, 128)
    k_c = Key("data.x", Observation, shape=ShapeContract(ndim=3))
    
    c1 = MockComponent("P", provides={k_p: WriteMode.NEW})
    c2 = MockComponent("C", requires={k_c})
    
    with pytest.raises(RuntimeError) as excinfo:
        BlackboardEngine(components=[c1, c2], device=torch.device("cpu"))
        
    error_msg = str(excinfo.value)
    assert "provides: shape (B, ?)" in error_msg
    assert "requires: shape (B, ?, ?)" in error_msg
    assert "Rank mismatch" in error_msg
