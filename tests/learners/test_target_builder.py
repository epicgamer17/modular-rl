import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from agents.learners.target_builder import BaseTargetBuilder

pytestmark = pytest.mark.unit


def test_base_target_builder_instantiation():
    """
    Test that BaseTargetBuilder cannot be instantiated directly.
    """
    with pytest.raises(
        TypeError, match="Can't instantiate abstract class BaseTargetBuilder"
    ):
        BaseTargetBuilder()


def test_dummy_target_builder_interface_and_return(dummy_target_builder):
    """
    Test that DummyTargetBuilder correctly implements the interface and returns the expected format.
    """
    torch.manual_seed(42)
    builder = dummy_target_builder

    # Create fake inputs
    batch = {"obs": torch.randn(2, 4)}
    predictions = {"values": torch.randn(2, 1)}
    network = nn.Sequential(nn.Linear(4, 1))

    # Call build_targets
    targets = builder.build_targets(batch, predictions, network)

    # Assert return type and content
    assert isinstance(targets, dict), "DummyTargetBuilder must return a dictionary"
    assert len(targets) == 0, "DummyTargetBuilder must return an empty dictionary"


def test_dummy_target_builder_signature_enforcement(dummy_target_builder):
    """
    Test that DummyTargetBuilder signature is strictly enforced using mocks.
    """
    builder = dummy_target_builder

    mock_batch = MagicMock(spec=dict)
    mock_predictions = MagicMock(spec=dict)
    mock_network = MagicMock(spec=nn.Module)

    # This should work without error
    targets = builder.build_targets(mock_batch, mock_predictions, mock_network)
    assert targets == {}

    # Test with missing arguments to ensure signature check (Python will raise TypeError)
    with pytest.raises(TypeError):
        builder.build_targets(mock_batch, mock_predictions)  # type: ignore
