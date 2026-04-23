import pytest
from core.schema import Schema, Field, TensorSpec, TAG_ON_POLICY

pytestmark = pytest.mark.unit

def test_schema_compatibility():
    """
    Test 1.2: Verify schema creation and compatibility logic.
    """
    # 1. create base schema
    obs_spec = TensorSpec(shape=(4,), dtype="float32", tags=[TAG_ON_POLICY])
    action_spec = TensorSpec(shape=(1,), dtype="int64")
    
    schema_a = Schema(fields=[
        Field("obs", obs_spec),
        Field("action", action_spec)
    ])
    
    # 2. identical schema -> valid
    schema_b = Schema(fields=[
        Field("obs", TensorSpec(shape=(4,), dtype="float32", tags=[TAG_ON_POLICY])),
        Field("action", TensorSpec(shape=(1,), dtype="int64"))
    ])
    assert schema_a.is_compatible(schema_b), "Identical schemas should be compatible"
    
    # 3. mismatched shapes -> invalid
    schema_c = Schema(fields=[
        Field("obs", TensorSpec(shape=(8,), dtype="float32")), # Mismatched shape
        Field("action", action_spec)
    ])
    assert not schema_a.is_compatible(schema_c), "Schemas with mismatched shapes should not be compatible"
    
    # 4. missing fields -> invalid
    schema_d = Schema(fields=[
        Field("obs", obs_spec)
        # Missing action
    ])
    assert not schema_a.is_compatible(schema_d), "Schemas with missing fields should not be compatible"

    # 5. mismatched dtype -> invalid
    schema_e = Schema(fields=[
        Field("obs", TensorSpec(shape=(4,), dtype="float64")), # Mismatched dtype
        Field("action", action_spec)
    ])
    assert not schema_a.is_compatible(schema_e), "Schemas with mismatched dtypes should not be compatible"

def test_duplicate_fields_fail():
    """Verify that schemas cannot have duplicate field names."""
    with pytest.raises(AssertionError, match="Duplicate field names"):
        Schema(fields=[
            Field("obs", TensorSpec(shape=(1,), dtype="float32")),
            Field("obs", TensorSpec(shape=(1,), dtype="float32"))
        ])

if __name__ == "__main__":
    # For direct execution
    test_schema_compatibility()
    test_duplicate_fields_fail()
    print("Test 1.2 Passed!")
