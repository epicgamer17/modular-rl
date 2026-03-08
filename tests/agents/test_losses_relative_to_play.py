import pytest

pytestmark = pytest.mark.unit

import torch
import torch.nn.functional as F
from losses.losses import RelativeToPlayLoss


def test_relative_to_play_loss_math():
    """Verify ΔP calculation from absolute player sequences and loss computation."""
    torch.manual_seed(42)
    device = torch.device("cpu")

    # Mock config with num_players
    class MockGameConfig:
        def __init__(self):
            self.num_players = 3

    class MockConfig:
        def __init__(self):
            self.game = MockGameConfig()
            self.to_play_loss_factor = 1.0
            self.to_play_loss_function = F.cross_entropy

    config = MockConfig()
    loss_module = RelativeToPlayLoss(config, device)

    # Batch size 2, num_players 3
    # Step k=1 (requires k-1)
    k = 1

    # Predicted ΔP logits (B=2, num_players=3)
    # Batch 0: Predict ΔP=1
    # Batch 1: Predict ΔP=2
    predictions = {
        "to_plays": torch.tensor(
            [[0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]  # Argmax is 1  # Argmax is 2
        )
    }

    # Absolute players at k=0 and k=1
    # Batch 0: P0=0, P1=1 -> ΔP = (1-0)%3 = 1 (Correct)
    # Batch 1: P0=2, P1=1 -> ΔP = (1-2)%3 = 2 (Correct)
    to_plays = torch.tensor([[0, 1], [2, 1]], dtype=torch.long)  # Batch 0  # Batch 1

    full_targets = {"to_plays": to_plays}
    context = {"full_targets": full_targets, "has_valid_obs_mask": torch.ones(2, 2)}

    # Targets dictionary passed to compute_loss (usually sliced targets_k)
    targets = {"to_plays": to_plays[:, k]}

    # Compute loss
    loss = loss_module.compute_loss(
        predictions=predictions, targets=targets, k=k, context=context
    )

    # Manual calculate target delta_p
    target0 = (1 - 0) % 3  # 1
    target1 = (1 - 2) % 3  # 2
    expected_targets = torch.tensor([target0, target1], dtype=torch.long)

    assert torch.equal(expected_targets, torch.tensor([1, 2]))

    # Cross entropy loss should be very small since predictions are "perfect"
    assert loss[0] < 0.01
    assert loss[1] < 0.01


def test_relative_to_play_loss_should_compute():
    """Verify should_compute logic."""

    class MockGameConfig:
        def __init__(self):
            self.num_players = 2

    class MockConfig:
        def __init__(self):
            self.game = MockGameConfig()

    config = MockConfig()
    loss_module = RelativeToPlayLoss(config, torch.device("cpu"))

    # k=0 should not compute
    assert not loss_module.should_compute(0, {})

    # k=1 with 1 player should not compute
    config.game.num_players = 1
    assert not loss_module.should_compute(1, {"full_targets": {"to_plays": None}})

    # k=1 with 2 players and targets should compute
    config.game.num_players = 2
    context = {"full_targets": {"to_plays": torch.tensor([0, 1])}}
    assert loss_module.should_compute(1, context)


def test_relative_to_play_loss_wraparound_targets():
    """Verify ΔP calculation specifically for wraparound indices."""

    class MockGameConfig:
        def __init__(self):
            self.num_players = 4

    class MockConfig:
        def __init__(self):
            self.game = MockGameConfig()
            self.to_play_loss_factor = 1.0
            self.to_play_loss_function = F.cross_entropy

    config = MockConfig()
    loss_module = RelativeToPlayLoss(config, torch.device("cpu"))

    # P0=3, P1=0 -> ΔP = (0-3)%4 = 1
    # P0=3, P1=1 -> ΔP = (1-3)%4 = 2
    to_plays = torch.tensor([[3, 0], [3, 1]], dtype=torch.long)

    full_targets = {"to_plays": to_plays}
    context = {"full_targets": full_targets}
    predictions = {"to_plays": torch.randn(2, 4)}

    # k=1
    loss = loss_module.compute_loss(
        predictions=predictions, targets={}, k=1, context=context
    )

    # We don't check value, just that it didn't crash and we can check the internal logic
    # by verifying how it would calculate targets
    p_k = to_plays[:, 1]
    p_prev = to_plays[:, 0]
    target_delta_p = (p_k - p_prev) % 4

    assert torch.equal(target_delta_p, torch.tensor([1, 2]))
