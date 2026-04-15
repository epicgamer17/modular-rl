import pytest
import torch
from core.contracts import Key, Observation, ShapeContract, check_shape_compatibility
from core.shape_validation import validate_tensor

pytestmark = pytest.mark.unit

def test_time_dim_compatibility_success():
    """Verifies that matching time_dim passes validation."""
    k_p = Key("data.x", Observation, shape=ShapeContract(time_dim=1))
    k_c = Key("data.x", Observation, shape=ShapeContract(time_dim=1))
    
    issues = check_shape_compatibility(k_p, k_c)
    assert not issues

def test_time_dim_compatibility_mismatch():
    """Verifies that mismatched time_dim fails validation."""
    k_p = Key("data.x", Observation, shape=ShapeContract(time_dim=1))
    k_c = Key("data.x", Observation, shape=ShapeContract(time_dim=2))
    
    issues = check_shape_compatibility(k_p, k_c)
    assert any("Time dimension position mismatch" in i for i in issues)

def test_time_dim_gap_provider_missing():
    """Verifies that if consumer requires time_dim but provider doesn't have it, it fails."""
    k_p = Key("data.x", Observation, shape=ShapeContract(time_dim=None))
    k_c = Key("data.x", Observation, shape=ShapeContract(time_dim=1))
    
    issues = check_shape_compatibility(k_p, k_c)
    assert any("Time dimension gap" in i for i in issues)

def test_time_dim_gap_consumer_missing():
    """Verifies that if provider has time_dim but consumer explicitly wants None, it fails."""
    k_p = Key("data.x", Observation, shape=ShapeContract(time_dim=1))
    k_c = Key("data.x", Observation, shape=ShapeContract(time_dim=None))
    
    # Wait, check_shape_compatibility only checks if consumer HAS a requirement.
    # If c.time_dim is None, it used to skip. 
    # But I updated it to check if p.time_dim is NOT None when c.time_dim IS None.
    issues = check_shape_compatibility(k_p, k_c)
    assert any("Time dimension mismatch: provider has a sequence dimension (T)" in i for i in issues)

def test_runtime_validation_success():
    """Verifies runtime validation of time_dim."""
    k = Key("data.x", Observation, shape=ShapeContract(time_dim=1))
    # [B, T, D] -> ndim=3
    val = torch.randn(2, 5, 10)
    validate_tensor(k, val)

def test_runtime_validation_failure():
    """Verifies runtime validation failure when tensor lacks enough dimensions for time_dim."""
    k = Key("data.x", Observation, shape=ShapeContract(time_dim=2))
    # [B, T] -> ndim=2. Contract requires sequence dim at index 2 (starting from 0).
    # So it needs at least 3 dimensions.
    val = torch.randn(2, 5)
    with pytest.raises(AssertionError, match="time_dim validation"):
        validate_tensor(k, val)
