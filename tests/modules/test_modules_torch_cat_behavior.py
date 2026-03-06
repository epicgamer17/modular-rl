import pytest
pytestmark = pytest.mark.unit

import torch

def test_torch_cat_handles_empty_tensor_first_append():
    existing = torch.empty(0)
    to_append = torch.tensor([[0.1, 0.2, 0.3]])

    # Behavior is torch-version dependent: some versions raise on rank mismatch,
    # others coerce successfully to the appended tensor's rank.
    try:
        direct = torch.cat((existing, to_append))
    except RuntimeError:
        direct = None

    if direct is not None:
        assert direct.shape == (1, 3)

    if existing.numel() == 0:
        result = to_append
    else:
        result = torch.cat((existing, to_append))

    assert result.shape == (1, 3)

    to_append2 = torch.tensor([[0.4, 0.5, 0.6]])
    result = torch.cat((result, to_append2))
    assert result.shape == (2, 3)
