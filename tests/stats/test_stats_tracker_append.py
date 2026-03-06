import pytest
pytestmark = pytest.mark.unit

import torch

from stats.stats import StatTracker


def test_append_scalar():
    """Test appending simple scalar values (backward compatibility)."""
    tracker = StatTracker("test_model")
    tracker.append("scalar_metric", 1.0)
    tracker.append("scalar_metric", 2.0)

    assert tracker.stats["scalar_metric"] == [1.0, 2.0]


def test_append_tensor_2d():
    """Test appending 2D tensors stores per-step entries."""
    tracker = StatTracker("test_model")
    tensor1 = torch.randn(1, 5)
    tensor2 = torch.randn(1, 5)

    tracker.append("tensor_metric", tensor1)
    tracker.append("tensor_metric", tensor2)

    stored = tracker.stats["tensor_metric"]
    assert len(stored) == 2
    assert torch.equal(stored[0], tensor1)
    assert torch.equal(stored[1], tensor2)


def test_append_mismatched_shapes():
    """Mismatched tensor shapes are stored independently at append-time."""
    tracker = StatTracker("test_model")
    tracker.append("mismatch_metric", torch.randn(1, 5))
    tracker.append("mismatch_metric", torch.randn(1, 3))
    stored = tracker.stats["mismatch_metric"]
    assert len(stored) == 2
    assert tuple(stored[0].shape) == (1, 5)
    assert tuple(stored[1].shape) == (1, 3)
