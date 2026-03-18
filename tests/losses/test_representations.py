import pytest
import torch
import torch.nn.functional as F
from losses.representations import (
    ScalarRepresentation,
    TwoHotRepresentation,
    CategoricalRepresentation,
    ClassificationRepresentation,
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
    vmin, vmax, bins = -300.0, 300.0, 601
    repr = TwoHotRepresentation(vmin=vmin, vmax=vmax, bins=bins)
    assert repr.num_classes == 601

    # Test to_representation (scalar -> distribution)
    targets = torch.tensor([15.5, -10.2])
    dist = repr.to_representation(targets)
    assert dist.shape == (2, 601)
    assert torch.allclose(dist.sum(dim=-1), torch.ones(2))

    # Test to_scalar (logits -> scalar)
    logits = dist * 100.0  # Large logits to make softmax close to one-hot
    recovered = repr.to_scalar(logits)
    assert torch.allclose(recovered, targets, atol=1e-2)


def test_classification_representation():
    num_classes = 10
    repr = ClassificationRepresentation(num_classes=num_classes)
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

    # Classification
    repr = get_representation(num_classes=5)
    assert isinstance(repr, ClassificationRepresentation)
    assert repr.num_classes == 5

    # Two-Hot
    repr = get_representation(vmin=-300, vmax=300, bins=601)
    assert isinstance(repr, TwoHotRepresentation)
    assert repr.num_classes == 601
