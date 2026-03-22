import pytest
import torch
import torch.nn.functional as F
from agents.learner.losses.representations import (
    ScalarRepresentation,
    DiscreteSupportRepresentation,
    C51Representation,
    ClassificationRepresentation,
    get_representation,
)

pytestmark = pytest.mark.unit


def test_scalar_representation():
    repr = ScalarRepresentation()
    targets = torch.tensor([1.0, 2.0, 3.0])
    rep = repr.to_representation(targets)
    assert rep.shape == (3, 1)
    assert torch.allclose(rep.squeeze(), targets)

    # test to_expected_value
    val_tensor = rep.unsqueeze(1) # [3, 1, 1]
    expected = repr.to_expected_value(val_tensor)
    assert expected.shape == (3, 1)
    assert torch.allclose(expected.squeeze(), targets)


def test_discrete_support_representation():
    # bins=11 -> range [0, 10]
    repr = DiscreteSupportRepresentation(vmin=0.0, vmax=10.0, bins=11)
    targets = torch.tensor([0.0, 5.0, 10.0, 5.5])
    
    rep = repr.to_representation(targets)
    assert rep.shape == (4, 11)
    
    # 0.0 should be [1, 0, 0, ...]
    assert rep[0, 0] == 1.0
    # 5.0 should be [0, 0, 0, 0, 0, 1, 0, ...]
    assert rep[1, 5] == 1.0
    # 10.0 should be [..., 0, 1]
    assert rep[2, 10] == 1.0
    # 5.5 should be [..., 0.5, 0.5, ...] at indices 5 and 6
    assert rep[3, 5] == 0.5
    assert rep[3, 6] == 0.5

    # test to_expected_value
    # Use log to create logits that won't saturate softmax
    logits = torch.log(rep + 1e-8).unsqueeze(1) # [4, 1, 11]
    expected = repr.to_expected_value(logits)
    assert torch.allclose(expected.squeeze(), targets, atol=1e-5)


def test_classification_representation():
    repr = ClassificationRepresentation(num_classes=4)
    # Target indices
    targets = torch.tensor([0, 1, 2, 3])
    rep = repr.to_representation(targets)
    assert rep.shape == (4, 4)
    assert torch.allclose(rep, torch.eye(4))

    # Expect logprob selector or argmax
    logits = torch.randn(2, 1, 4)
    expected = repr.to_expected_value(logits)
    assert expected.shape == (2, 1)


def test_get_representation():
    # get_representation(config: Optional[Union[Dict[str, Any], int, Any]] = None, **kwargs)
    r1 = get_representation(type="scalar")
    assert isinstance(r1, ScalarRepresentation)
    
    config = {
        'vmin': -10,
        'vmax': 10,
        'bins': 21
    }
    r2 = get_representation(config, type="discrete_support")
    assert isinstance(r2, DiscreteSupportRepresentation)
    
    r3 = get_representation(num_classes=4)
    assert isinstance(r3, ClassificationRepresentation)
    assert r3._num_classes == 4
