import pytest
import math
from search.search_py.min_max_stats import MinMaxStats

pytestmark = pytest.mark.unit

def test_min_max_stats_unbounded_normalization():
    """MATH: MinMaxStats must correctly track expanding bounds and normalize to [0,1]."""
    stats = MinMaxStats(known_bounds=None)
    
    # 1. Initial state: unknown bounds should return clamped value (0.0)
    # CONTRACT: not to output raw values
    assert stats.normalize(5.0) == 0.0
    
    # 2. Update with new bounds
    stats.update(5.0)
    stats.update(10.0)
    
    # 3. Normalization checks
    # Math: (7.5 - 5) / (10 - 5) = 2.5 / 5.0 = 0.5
    assert stats.normalize(7.5) == 0.5
    assert stats.normalize(5.0) == 0.0
    assert stats.normalize(10.0) == 1.0

def test_min_max_stats_strict_known_bounds():
    """CONTRACT: If bounds are known, normalization uses them immediately."""
    stats = MinMaxStats(known_bounds=[-1.0, 1.0])
    
    # Math: (0.5 - (-1)) / (1 - (-1)) = 1.5 / 2.0 = 0.75
    assert stats.normalize(0.5) == 0.75
    
    # Updating within bounds should not shift the normalization
    stats.update(0.0)
    assert stats.normalize(0.5) == 0.75
