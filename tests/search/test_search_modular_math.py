import pytest
import torch
from search.search_py.utils import _safe_log_prob

pytestmark = pytest.mark.unit


def test_modular_search_safe_log_prob():
    # Enforce strict determinism
    torch.manual_seed(42)

    # Create a probability distribution with a definite 0.0 value
    probs = torch.tensor([0.0, 0.5, 1.0])

    # Process it through the search tree's static math handler
    log_prob = _safe_log_prob(probs)

    # The 0.0 probability MUST map perfectly to -inf
    assert log_prob[0].item() == -float("inf")

    # The >0.0 probabilities must map to exact logarithms
    assert torch.isclose(log_prob[1], torch.tensor(0.5).log())
    assert torch.isclose(log_prob[2], torch.tensor(0.0))  # log(1.0) == 0.0
