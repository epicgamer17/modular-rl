import pytest
import math

try:
    import search
    search.set_backend("cpp")
    from search.search_cpp import MinMaxStats
    HAS_CPP = True
except (ImportError, RuntimeError):
    HAS_CPP = False

pytestmark = pytest.mark.unit

@pytest.mark.skipif(not HAS_CPP, reason="C++ search backend not available.")
def test_min_max_stats_unbounded_normalization():
    """MATH: C++ MinMaxStats must correctly track expanding bounds and normalize to [0,1]."""
    # Initialise with empty list for unknown bounds
    stats = MinMaxStats([], False, 0.01)
    
    # 1. Initial state: unknown bounds should return clamped value (0.0)
    # CONTRACT: not to output raw values
    assert stats.normalize(5.0) == 0.0
    
    # 2. Update with new bounds
    stats.update(5.0)
    stats.update(10.0)
    
    # 3. Normalization checks
    # Math: (7.5 - 5) / (10 - 5) = 2.5 / 5.0 = 0.5
    assert math.isclose(stats.normalize(7.5), 0.5, rel_tol=1e-7)
    assert math.isclose(stats.normalize(5.0), 0.0, rel_tol=1e-7)
    assert math.isclose(stats.normalize(10.0), 1.0, rel_tol=1e-7)

@pytest.mark.skipif(not HAS_CPP, reason="C++ search backend not available.")
def test_min_max_stats_strict_known_bounds():
    """CONTRACT: If bounds are known, normalization uses them immediately."""
    stats = MinMaxStats([-1.0, 1.0], False, 0.01)
    
    # Math: (0.5 - (-1)) / (1 - (-1)) = 1.5 / 2.0 = 0.75
    assert math.isclose(stats.normalize(0.5), 0.75, rel_tol=1e-7)
    
    # Updating within bounds should not shift the normalization
    stats.update(0.0)
    assert math.isclose(stats.normalize(0.5), 0.75, rel_tol=1e-7)

@pytest.mark.skipif(not HAS_CPP, reason="C++ search backend not available.")
def test_min_max_stats_known_bounds_apply_on_first_call():
    """CONTRACT: C++ MinMaxStats must use known bounds on the first normalize call."""
    stats = MinMaxStats([-1.0, 1.0], False, 0.01)

    assert math.isclose(stats.normalize(-1.0), 0.0, rel_tol=1e-7)
    assert math.isclose(stats.normalize(0.0), 0.5, rel_tol=1e-7)
    assert math.isclose(stats.normalize(1.0), 1.0, rel_tol=1e-7)
