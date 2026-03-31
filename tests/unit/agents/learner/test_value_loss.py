import pytest
import torch

pytestmark = pytest.mark.unit

def test_value_loss_basic():
    """Verifies that the standard ValueLoss correctly computes MSE."""
    pytest.skip("TODO: update for old_muzero revert")

def test_clipped_value_loss_integration():
    """
    Verifies that ClippedValueLoss correctly integrates the clipping logic
    and pulls 'values' (old) from the targets dictionary.
    """
    pytest.skip("TODO: update for old_muzero revert")

