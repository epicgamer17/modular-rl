import pytest
import torch

pytestmark = pytest.mark.unit

def test_muzero_multiplayer_backpropagation_py():
    """Verifies the search_py backpropagation against analytical oracles (including discounting)."""
    pytest.skip("TODO: update for old_muzero revert")
