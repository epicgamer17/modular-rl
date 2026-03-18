import pytest
import torch
import torch.nn.functional as F
from losses.losses import (
    ScalarRepresentation,
    TwoHotRepresentation,
    CategoricalRepresentation,
    get_representation,
)

pytestmark = pytest.mark.unit


def test_scalar_representation():
    repr = ScalarRepresentation()
    assert repr.num_classes == 1

    logits = torch.tensor([[10.5], [-2.3]])
    scalar = repr.to_scalar(logits)
    assert torch.allclose(scalar, torch.tensor([10.5, -2.3]))

    targets = torch.tensor([5.0, -1.0])
    representation = repr.to_representation(targets)
    assert torch.allclose(representation, targets)


def test_two_hot_representation():
    support_range = 300
    repr = TwoHotRepresentation(support_range=support_range)
    assert repr.num_classes == 601

    # Test to_representation (scalar -> distribution)
    targets = torch.tensor([15.5, -10.2])
    dist = repr.to_representation(targets)
    assert dist.shape == (2, 601)
    assert torch.allclose(dist.sum(dim=-1), torch.ones(2))

    # Test to_scalar (logits -> scalar)
    # If we pass the distribution as logits (large values for the two-hot bins)
    # to_scalar should recover something close to the original targets
    logits = dist * 100.0  # Large logits to make softmax close to one-hot
    recovered = repr.to_scalar(logits)
    assert torch.allclose(recovered, targets, atol=1e-2)


def test_categorical_representation():
    num_classes = 10
    repr = CategoricalRepresentation(num_classes=num_classes)
    assert repr.num_classes == 10

    # Test to_representation (index -> one-hot)
    targets = torch.tensor([3, 7])
    one_hot = repr.to_representation(targets)
    assert one_hot.shape == (2, 10)
    assert one_hot[0, 3] == 1.0
    assert one_hot[1, 7] == 1.0
    assert one_hot.sum() == 2.0

    # Test to_scalar (logits -> index)
    logits = torch.zeros((2, 10))
    logits[0, 5] = 10.0
    logits[1, 2] = 10.0
    indices = repr.to_scalar(logits)
    assert torch.allclose(indices, torch.tensor([5.0, 2.0]))


def test_get_representation_factory():
    # Scalar
    repr = get_representation(num_classes=1)
    assert isinstance(repr, ScalarRepresentation)

    # Categorical
    repr = get_representation(num_classes=5)
    assert isinstance(repr, CategoricalRepresentation)
    assert repr.num_classes == 5

    # Two-Hot
    repr = get_representation(support_range=300)
    assert isinstance(repr, TwoHotRepresentation)
    assert repr.num_classes == 601
