import pytest
import torch
import numpy as np
from agents.learner.losses.shape_validator import ShapeValidator

pytestmark = pytest.mark.unit


def test_shape_validator_muzero_valid(muzero_config):
    """
    Test that ShapeValidator accepts valid MuZero sequence shapes.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    validator = ShapeValidator(muzero_config)
    B = muzero_config.minibatch_size
    T = muzero_config.unroll_steps + 1
    A = muzero_config.game.num_actions

    predictions = {
        "policies": torch.randn(B, T, A),
        "values": torch.randn(B, T, validator.atom_size),
    }
    targets = {
        "policies": torch.randn(B, T, A),
        "values": torch.randn(B, T, validator.atom_size),
    }

    # Should not raise
    validator.validate(predictions, targets)


def test_shape_validator_ppo_valid(ppo_config):
    """
    Test that ShapeValidator accepts valid PPO single-step shapes (T=1 omitted).
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # Ensure T=1 for PPO-style check
    ppo_config.unroll_steps = 0
    validator = ShapeValidator(ppo_config)
    B = ppo_config.minibatch_size
    A = ppo_config.game.num_actions

    predictions = {
        "policies": torch.randn(B, A),
        "values": torch.randn(B),
    }
    targets = {
        "policies": torch.randn(B, A),
        "values": torch.randn(B),
    }

    # Should not raise
    validator.validate(predictions, targets)


def test_shape_validator_mismatch_raises(muzero_config):
    """
    Test that ShapeValidator raises AssertionError on various shape mismatches.
    """
    torch.manual_seed(42)

    validator = ShapeValidator(muzero_config)
    B = muzero_config.minibatch_size
    T = muzero_config.unroll_steps + 1
    A = muzero_config.game.num_actions

    # 1. Batch size mismatch
    predictions = {"policies": torch.randn(B + 1, T, A)}
    with pytest.raises(AssertionError, match="batch size mismatch"):
        validator.validate(predictions, {})

    # 2. Sequence length mismatch
    predictions = {"policies": torch.randn(B, T + 1, A)}
    with pytest.raises(AssertionError, match="sequence length mismatch"):
        validator.validate(predictions, {})

    # 3. Action dimension mismatch
    predictions = {"policies": torch.randn(B, T, A + 1)}
    with pytest.raises(AssertionError, match="action dim mismatch"):
        validator.validate(predictions, {})


def test_shape_validator_transition_data(muzero_config):
    """
    Test that ShapeValidator accepts transition-aligned data (e.g. rewards) with T-1 sequence length.
    """
    torch.manual_seed(42)

    validator = ShapeValidator(muzero_config)
    B = muzero_config.minibatch_size
    T_minus_1 = muzero_config.unroll_steps

    targets = {
        "rewards": torch.randn(B, T_minus_1),
    }

    # Should not raise
    validator.validate({}, targets)
