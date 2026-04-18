"""
Regression test for NStepUnrollProcessor reward and to-play masking.

Before the fix, reward_mask and to_play_mask were derived from dynamics_mask
(obs_mask & ~raw_dones), which excluded terminal states. This caused the
terminal reward (e.g., win=1.0 in TicTacToe) to always be masked out during
loss computation, making the reward loss converge to 0 trivially.

The fix uses same_game mask (which includes terminal states) for both
reward_mask and to_play_mask, matching the old pre-refactor behavior from
agents/learner/target_builders.py::SequenceMaskBuilder.
"""
import numpy as np
import pytest
import torch

from data.ingestion import Sequence
from registries import make_muzero_replay_buffer

pytestmark = pytest.mark.unit


@pytest.fixture
def filled_buffer():
    """Creates a buffer with episodes that have terminal rewards."""
    torch.manual_seed(42)
    np.random.seed(42)

    na = 9
    od = (9, 3, 3)
    buf = make_muzero_replay_buffer(
        obs_dim=od,
        num_actions=na,
        buffer_size=1000,
        batch_size=8,
        unroll_steps=5,
    )

    for ep in range(20):
        seq = Sequence(num_players=2)
        seq.append(
            observation=np.random.rand(*od).astype(np.float32),
            terminated=False,
            truncated=False,
            reward=0.0,
            value=0.5,
            policy=np.ones(na, dtype=np.float32) / na,
            player_id=0,
            legal_moves=list(range(na)),
        )
        game_len = np.random.randint(5, 9)
        for i in range(game_len):
            is_last = i == game_len - 1
            # Next player
            next_player = (i + 1) % 2
            seq.append(
                observation=np.random.rand(*od).astype(np.float32),
                terminated=is_last,
                truncated=False,
                action=i % na,
                reward=1.0 if is_last else 0.0,
                policy=np.ones(na, dtype=np.float32) / na,
                value=0.5 if not is_last else 0.0,
                player_id=next_player,
                legal_moves=list(range(na)),
            )
        buf.store_aggregate(seq)

    return buf


def test_terminal_reward_not_masked(filled_buffer):
    """Terminal rewards must be visible through the reward mask."""
    batch = filled_buffer.sample()

    rewards = batch["rewards"]
    reward_mask = batch["reward_mask"]

    non_zero_rewards = rewards[rewards != 0]
    assert non_zero_rewards.numel() > 0, "No non-zero rewards in batch"

    masked_rewards = rewards[reward_mask]
    non_zero_masked = masked_rewards[masked_rewards != 0]
    assert non_zero_masked.numel() > 0, (
        "All non-zero rewards are masked out. "
        "Terminal rewards should be included in reward_mask."
    )


def test_reward_mask_excludes_root(filled_buffer):
    """Reward mask must be False at position 0 (root has no incoming reward)."""
    batch = filled_buffer.sample()
    assert not batch["reward_mask"][:, 0].any(), (
        "reward_mask should be False at root (position 0)"
    )


def test_to_play_mask_excludes_root(filled_buffer):
    """To-play mask must be False at position 0."""
    batch = filled_buffer.sample()
    assert not batch["to_play_mask"][:, 0].any(), (
        "to_play_mask should be False at root (position 0)"
    )


def test_to_play_mask_includes_terminal(filled_buffer):
    """To-play mask should include terminal states (whose turn it was is valid)."""
    batch = filled_buffer.sample()

    to_play_mask = batch["to_play_mask"]
    dones = batch["dones"]

    # Find positions where done=True and it's within the same game
    terminal_positions = dones & batch["is_same_episode"]
    # Exclude root (position 0)
    terminal_positions[:, 0] = False

    if terminal_positions.any():
        # At least some terminal positions should be masked in
        terminal_masked_in = to_play_mask[terminal_positions]
        assert terminal_masked_in.any(), (
            "No terminal states are included in to_play_mask. "
            "Terminal states have valid to-play information."
        )


def test_reward_and_to_play_masks_match(filled_buffer):
    """Reward mask and to-play mask should be identical."""
    batch = filled_buffer.sample()
    torch.testing.assert_close(
        batch["reward_mask"],
        batch["to_play_mask"],
        msg="reward_mask and to_play_mask should be identical",
    )
