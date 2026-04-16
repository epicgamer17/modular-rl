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
    """Verifies the new structured error message for semantic shape mismatch."""
    # Component A provides (B, T, 128)
    k_p = Key("data.x", Observation, shape=ShapeContract(
        semantic_shape=("B", "T", "A"),
        event_shape=(128,),
    ))
    
    # Component B requires (B, 128)
    k_c = Key("data.x", Observation, shape=ShapeContract(
        semantic_shape=("B", "A"),
        event_shape=(128,),
    ))
    
    c1 = MockComponent("Producer", provides={k_p: WriteMode.NEW})
    c2 = MockComponent("Consumer", requires={k_c})
    
    with pytest.raises(RuntimeError) as excinfo:
        BlackboardEngine(components=[c1, c2], device=torch.device("cpu"))
    
    error_msg = str(excinfo.value)
    
    # Check for structured parts
    assert "Component A (Key: data.x, Provider: [0] MockComponent)" in error_msg
    assert "provides: shape (B, T, 128)" in error_msg
    assert "Component B (Key: data.x, Consumer: [1] MockComponent)" in error_msg
    assert "requires: shape (B, 128)" in error_msg
    assert "Error:" in error_msg
    assert "Rank mismatch" in error_msg

def test_better_error_message_rank_mismatch():
    """Verifies the new structured error message for rank mismatch."""
    # Component A provides (B, A)
    k_p = Key("data.x", Observation, shape=ShapeContract(semantic_shape=("B", "A")))
    
    # Component B requires (B, T, A)
    k_c = Key("data.x", Observation, shape=ShapeContract(semantic_shape=("B", "T", "A")))
    
    c1 = MockComponent("P", provides={k_p: WriteMode.NEW})
    c2 = MockComponent("C", requires={k_c})
    
    with pytest.raises(RuntimeError) as excinfo:
        BlackboardEngine(components=[c1, c2], device=torch.device("cpu"))
        
    error_msg = str(excinfo.value)
    assert "provides: shape (B, A)" in error_msg
    assert "requires: shape (B, T, A)" in error_msg
    assert "Rank mismatch" in error_msg

