from core.contracts import Key, ShapeContract, check_shape_compatibility, ValueEstimate, Scalar

def test_star_event_shape_overlap():
    # Consumer has a wildcard and an event axis
    c_key = Key("test", ValueEstimate[Scalar], shape=ShapeContract(
        semantic_shape=("B", "*", "F"), event_shape=(64,)
    ))
    # Provider has two event axes
    p_key = Key("test", ValueEstimate[Scalar], shape=ShapeContract(
        semantic_shape=("B", "H", "F"), event_shape=(128, 64)
    ))
    
    # This should logically succeed if "*" is a wildcard.
    # But event_shape lengths differ (1 vs 2).
    issues = check_shape_compatibility(p_key, c_key)
    print(f"Mix check: {issues}")

if __name__ == "__main__":
    test_star_event_shape_overlap()
