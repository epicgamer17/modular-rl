import pytest
import torch
import torch.nn.functional as F
from modules.heads.continuation import ContinuationHead
from configs.modules.architecture_config import ArchitectureConfig
from agents.learner.losses.representations import ClassificationRepresentation, ScalarRepresentation

pytestmark = pytest.mark.unit


def test_continuation_head_default_scalar():
    """Verifies default initialization to ScalarRepresentation when none is provided."""
    torch.manual_seed(42)
    arch_config = ArchitectureConfig({"noisy_sigma": 0.0})

    head = ContinuationHead(arch_config=arch_config, input_shape=(8,))
    assert isinstance(head.representation, ScalarRepresentation)

    x = torch.randn(2, 8)
    logits, state, continuation = head(x)
    # ScalarRepresentation squeezes the final dimension when num_bins=1
    assert continuation.shape == (2,)


def test_continuation_head_categorical_probability():
    """Verifies the head correctly extracts the class 1 probability for categorical ends."""
    torch.manual_seed(42)
    arch_config = ArchitectureConfig({"noisy_sigma": 0.0})
    representation = ClassificationRepresentation(num_classes=2)

    head = ContinuationHead(
        arch_config=arch_config, input_shape=(8,), representation=representation
    )

    x = torch.randn(2, 8)
    logits, state, continuation = head(x)
    expected_probs = F.softmax(logits, dim=-1)[..., 1]

    assert torch.allclose(continuation, expected_probs)
