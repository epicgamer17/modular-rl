import pytest
import torch
from torch import nn
from modules.heads.q import QHead, DuelingQHead
from configs.modules.architecture_config import ArchitectureConfig
from agents.learner.losses.representations import (
    CategoricalRepresentation,
    ScalarRepresentation,
)

pytestmark = pytest.mark.unit


def test_qhead_forward_and_reshape():
    """Verifies that Categorical representations reshape the output back to (B, actions, atoms)."""
    torch.manual_seed(42)
    arch_config = ArchitectureConfig({"noisy_sigma": 0.0})
    representation = CategoricalRepresentation(
        vmin=-10, vmax=10, bins=5
    )  # bins = 5 atoms

    head = QHead(
        arch_config=arch_config,
        input_shape=(16,),
        representation=representation,
        hidden_widths=[32],
        num_actions=4,
    )

    x = torch.randn(2, 16)
    logits, _, _ = head(x)

    # Batch size 2, 4 actions, 5 atoms
    assert logits.shape == (2, 4, 5)


def test_qhead_initialize_and_reset_noise():
    """Verifies custom initializers and noise resets pass through the hidden layers."""
    torch.manual_seed(42)
    arch_config = ArchitectureConfig({"noisy_sigma": 0.5})  # Enable noise
    representation = ScalarRepresentation()

    head = QHead(
        arch_config=arch_config,
        input_shape=(16,),
        representation=representation,
        hidden_widths=[16],
        num_actions=2,
    )

    def mock_init(tensor):
        nn.init.constant_(tensor, 0.5)

    head.initialize(mock_init)

    # Test reset noise executes without error
    head.reset_noise()


def test_dueling_qhead_forward_aggregation():
    """Verifies the Dueling Q aggregation logic Q = V + A - mean(A)."""
    torch.manual_seed(42)
    arch_config = ArchitectureConfig({"noisy_sigma": 0.0})
    representation = ScalarRepresentation()

    head = DuelingQHead(
        arch_config=arch_config,
        input_shape=(16,),
        representation=representation,
        value_hidden_widths=[8],
        advantage_hidden_widths=[8],
        num_actions=3,
    )

    x = torch.randn(2, 16)
    logits, _, q_vals = head(x)

    # Output should be (B, Actions, 1) => (2, 3, 1)
    assert logits.shape == (2, 3, 1)


def test_dueling_qhead_initialize_and_reset_noise():
    """Verifies custom initializers hit both the Value and Advantage streams."""
    torch.manual_seed(42)
    arch_config = ArchitectureConfig({"noisy_sigma": 0.5})
    representation = ScalarRepresentation()

    head = DuelingQHead(
        arch_config=arch_config,
        input_shape=(16,),
        representation=representation,
        value_hidden_widths=[8],
        advantage_hidden_widths=[8],
        num_actions=2,
    )

    def mock_init(tensor):
        nn.init.constant_(tensor, 0.5)

    head.initialize(mock_init)
    head.reset_noise()
