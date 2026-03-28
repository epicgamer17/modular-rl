import pytest
import torch
from search.aos_search.min_max_stats import VectorizedMinMaxStats, StaticMinMaxStats

pytestmark = pytest.mark.unit

def test_min_max_stats_unbounded_normalization():
    """MATH: VectorizedMinMaxStats must correctly track expanding bounds and normalize to [0,1]."""
    device = torch.device("cpu")
    # Initialise for batch size 1
    stats = VectorizedMinMaxStats.allocate(batch_size=1, device=device, known_bounds=None)
    
    # 1. Initial state: unknown bounds should return clamped value (0.0)
    q_initial = torch.tensor([5.0], dtype=torch.float32)
    assert stats.normalize(q_initial).item() == 0.0
    
    # 2. Update with new bounds
    # update(new_q_values, valid_mask)
    stats.update(torch.tensor([5.0], dtype=torch.float32), torch.tensor([True]))
    stats.update(torch.tensor([10.0], dtype=torch.float32), torch.tensor([True]))
    
    # 3. Normalization checks
    # Math: (7.5 - 5) / (10 - 5) = 2.5 / 5.0 = 0.5
    q_mid = torch.tensor([7.5], dtype=torch.float32)
    assert stats.normalize(q_mid).item() == 0.5
    
    assert stats.normalize(torch.tensor([5.0], dtype=torch.float32)).item() == 0.0
    assert stats.normalize(torch.tensor([10.0], dtype=torch.float32)).item() == 1.0

def test_min_max_stats_strict_known_bounds():
    """CONTRACT: StaticMinMaxStats uses fixed bounds immediately."""
    stats = StaticMinMaxStats(minimum=-1.0, maximum=1.0)
    
    # It should normalize immediately without needing .update() calls
    # Math: (0.5 - (-1)) / (1 - (-1)) = 1.5 / 2.0 = 0.75
    q = torch.tensor([0.5], dtype=torch.float32)
    assert stats.normalize(q).item() == 0.75
    
    # Updating (if we called it, though it's a no-op) should not shift the normalization
    stats.update(torch.tensor([0.0], dtype=torch.float32), torch.tensor([True]))
    assert stats.normalize(q).item() == 0.75

def test_vectorized_normalization_batching():
    """Verify separate bounds for separate batch elements."""
    device = torch.device("cpu")
    stats = VectorizedMinMaxStats.allocate(batch_size=2, device=device)
    
    # Batch 0: range [0, 10]
    # Batch 1: range [0, 100]
    stats.update(torch.tensor([0.0, 0.0], dtype=torch.float32), torch.tensor([True, True]))
    stats.update(torch.tensor([10.0, 100.0], dtype=torch.float32), torch.tensor([True, True]))
    
    q = torch.tensor([5.0, 5.0], dtype=torch.float32)
    norm = stats.normalize(q)
    
    # Batch 0: (5-0)/(10-0) = 0.5
    # Batch 1: (5-0)/(100-0) = 0.05
    assert norm[0].item() == pytest.approx(0.5)
    assert norm[1].item() == pytest.approx(0.05)
