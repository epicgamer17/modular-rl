import torch
from core.contracts import Key, ShapeContract, check_shape_compatibility, ValueEstimate, Scalar

def test_broadcasting_strictness():
    # Case 1: Same label "F", different sizes. Currently allowed by is_scalar_provider.
    # User wants this to FAIL.
    c_key = Key("test", ValueEstimate[Scalar], shape=ShapeContract(
        semantic_shape=("B", "F"), event_shape=(64,)
    ))
    p_key = Key("test", ValueEstimate[Scalar], shape=ShapeContract(
        semantic_shape=("B", "F"), event_shape=(1,)
    ))
    
    issues = check_shape_compatibility(p_key, c_key)
    print(f"Case 1 (F=64 vs F=1): {issues}")
    
    # Case 2: Consumer says "*", provider says "F=64".
    # This should be ALLOWED because "*" is a wildcard.
    # Currently it might fail because of event_shape length mismatch.
    c_key_star = Key("test", ValueEstimate[Scalar], shape=ShapeContract(
        semantic_shape=("B", "*") # event_shape is ()
    ))
    p_key_f64 = Key("test", ValueEstimate[Scalar], shape=ShapeContract(
        semantic_shape=("B", "F"), event_shape=(64,)
    ))
    
    issues_star = check_shape_compatibility(p_key_f64, c_key_star)
    print(f"Case 2 (* vs F=64): {issues_star}")

    # Case 3: Consumer says "*", provider says "F=1".
    # Should be ALLOWED.
    p_key_f1 = Key("test", ValueEstimate[Scalar], shape=ShapeContract(
        semantic_shape=("B", "F"), event_shape=(1,)
    ))
    issues_star_f1 = check_shape_compatibility(p_key_f1, c_key_star)
    print(f"Case 3 (* vs F=1): {issues_star_f1}")

if __name__ == "__main__":
    test_broadcasting_strictness()
