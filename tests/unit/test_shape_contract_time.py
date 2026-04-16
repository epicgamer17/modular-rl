import pytest
import torch
from core.contracts import Key, Observation, ShapeContract, check_shape_compatibility
from core.shape_validation import validate_tensor

pytestmark = pytest.mark.unit

def test_time_dim_compatibility_success():
    """Verifies that matching semantic_shape with 'T' passes validation."""
    k_p = Key("data.x", Observation, shape=ShapeContract(semantic_shape=("B", "T", "A")))
    k_c = Key("data.x", Observation, shape=ShapeContract(semantic_shape=("B", "T", "A")))
    
    issues = check_shape_compatibility(k_p, k_c)
    assert not issues

def test_time_dim_compatibility_mismatch():
    """Verifies that mismatched semantic_shape fails validation."""
    k_p = Key("data.x", Observation, shape=ShapeContract(semantic_shape=("B", "T", "A")))
    k_c = Key("data.x", Observation, shape=ShapeContract(semantic_shape=("B", "A")))
    
    issues = check_shape_compatibility(k_p, k_c)
    assert any("Rank mismatch" in i for i in issues)

def test_time_dim_gap_provider_missing():
    """Verifies that if consumer requires 'T' but provider doesn't have it, it fails."""
    k_p = Key("data.x", Observation, shape=ShapeContract(semantic_shape=("B", "A")))
    k_c = Key("data.x", Observation, shape=ShapeContract(semantic_shape=("B", "T", "A")))
    
    issues = check_shape_compatibility(k_p, k_c)
    assert any("Semantic axis mismatch" in i or "Rank mismatch" in i for i in issues)

def test_time_dim_gap_consumer_missing():
    """Verifies that if provider has 'T' but consumer explicitly wants a shape without it, it fails."""
    k_p = Key("data.x", Observation, shape=ShapeContract(semantic_shape=("B", "T", "A")))
    k_c = Key("data.x", Observation, shape=ShapeContract(semantic_shape=("B", "A")))
    
    issues = check_shape_compatibility(k_p, k_c)
    assert any("Rank mismatch" in i for i in issues)

def test_runtime_validation_success():
    """Verifies runtime validation of semantic_shape."""
    k = Key("data.x", Observation, shape=ShapeContract(semantic_shape=("B", "T", "A")))
    # [B, T, D] -> rank=3 matches
    val = torch.randn(2, 5, 10)
    validate_tensor(k, val)

def test_runtime_validation_failure():
    """Verifies runtime validation failure when tensor rank doesn't match semantic_shape."""
    k = Key("data.x", Observation, shape=ShapeContract(semantic_shape=("B", "T", "A")))
    # [B, T] -> rank=2. Contract requires rank 3.
    val = torch.randn(2, 5)
    with pytest.raises(AssertionError, match="rank mismatch"):
        validate_tensor(k, val)

