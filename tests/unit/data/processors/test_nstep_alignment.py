import numpy as np
import pytest
import torch
from data.processors.nstep import NStepUnrollProcessor

pytestmark = pytest.mark.unit


def test_nstep_return_alignment():
    """
    Verifies that N-step returns (Gt) are correctly calculated and aligned.

    Sequence:
        s0 -> r1=10 -> s1 -> r2=20 -> s2 (done)

    With n_step=3, gamma=1.0:
        G0 = r1 + r2 = 10 + 20 = 30
        G1 = r2 = 20
        G2 = 0 (Terminal)
    """
    unroll_steps = 2
    n_step = 3
    processor = NStepUnrollProcessor(
        unroll_steps=unroll_steps,
        n_step=n_step,
        gamma=1.0,
        num_actions=2,
        num_players=1,
        max_size=100,
    )

    seq_len = 10
    buffers = {
        "observations": torch.zeros(seq_len, 1),
        "actions": torch.zeros(seq_len, dtype=torch.int64),
        "rewards": torch.tensor([10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "values": torch.zeros(seq_len),
        "policies": torch.zeros(seq_len, 2),
        "player_ids": torch.zeros(seq_len, dtype=torch.int64),
        "to_plays": torch.zeros(seq_len),
        "chances": torch.zeros(seq_len, 1),
        "episode_id": torch.ones(seq_len, dtype=torch.int64),
        "legal_masks": torch.ones(seq_len, 2, dtype=torch.bool),
        "terminated": torch.tensor(
            [False, False, False, True, False, False, False, False, False, False]
        ),
        "truncated": torch.zeros(seq_len, dtype=torch.bool),
        "done": torch.tensor(
            [False, False, False, True, False, False, False, False, False, False]
        ),
        "is_same_episode": torch.ones(seq_len, dtype=torch.bool),
        "has_valid_obs_mask": torch.ones(seq_len, dtype=torch.bool),
        "ids": torch.arange(seq_len, dtype=torch.int64),
        "training_steps": torch.zeros(seq_len, dtype=torch.int64),
    }

    indices = [0]
    batch = processor.process_batch(indices, buffers)
    target_values = batch["values"]

    expected_values = torch.tensor([[30.0, 20.0, 0.0]])
    torch.testing.assert_close(
        target_values, expected_values, msg="N-step returns are misaligned!"
    )


def test_nstep_reward_alignment():
    """
    Verifies that rewards are correctly aligned with unroll steps.
    """
    unroll_steps = 10
    n_step = 1
    max_size = 100
    processor = NStepUnrollProcessor(
        unroll_steps=unroll_steps,
        n_step=n_step,
        gamma=1.0,
        num_actions=2,
        num_players=2,
        max_size=max_size,
    )

    seq_len = 12
    buffers = {
        "observations": torch.zeros(seq_len, 1),
        "actions": torch.zeros(seq_len, dtype=torch.int64),
        "rewards": torch.tensor(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 0.0, 0.0]
        ),
        "values": torch.zeros(seq_len),
        "policies": torch.zeros(seq_len, 2),
        "player_ids": torch.zeros(seq_len, dtype=torch.int64),
        "to_plays": torch.zeros(seq_len),
        "chances": torch.zeros(seq_len, 1),
        "episode_id": torch.ones(seq_len, dtype=torch.int64),
        "legal_masks": torch.ones(seq_len, 2, dtype=torch.bool),
        "terminated": torch.zeros(seq_len, dtype=torch.bool),
        "truncated": torch.zeros(seq_len, dtype=torch.bool),
        "done": torch.zeros(seq_len, dtype=torch.bool),
        "is_same_episode": torch.ones(seq_len, dtype=torch.bool),
        "has_valid_obs_mask": torch.ones(seq_len, dtype=torch.bool),
        "ids": torch.arange(seq_len, dtype=torch.int64),
        "training_steps": torch.zeros(seq_len, dtype=torch.int64),
    }

    indices = [0]
    batch = processor.process_batch(indices, buffers)
    rewards = batch["rewards"]
    # unroll_steps=10. target_rewards size 11.
    # raw_rewards: [10, 20, 30, ..., 100, 0]
    # targets: [0, r1, r2, ..., r10]
    expected_rewards = torch.tensor([[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]])
    torch.testing.assert_close(rewards, expected_rewards, msg="Rewards are misaligned!")


def test_nstep_multiplayer_sign_flipping():
    """
    Verifies that rewards are flipped correctly in a multiplayer zero-sum game.

    Sequence:
        s0 (p0 to play) -> r1=10 -> s1 (p1 to play) -> r2=5 -> s2 (p0 to play) -> r3=7

    G0 (p0's perspective, N=3):
        r1 (by p0) -> +10
        r2 (by p1) -> -5
        r3 (by p0) -> +7
        G0 = 10 - 5 + 7 = 12

    G1 (p1's perspective, N=3):
        r2 (by p1) -> +5
        r3 (by p0) -> -7
        r4 (0)     -> 0
        G1 = 5 - 7 = -2

    G2 (p0's perspective, N=3):
        r3 (by p0) -> +7
        r4 (0)     -> 0
        r5 (0)     -> 0
        G2 = 7
    """
    unroll_steps = 2
    n_step = 3
    processor = NStepUnrollProcessor(
        unroll_steps=unroll_steps,
        n_step=n_step,
        gamma=1.0,
        num_actions=2,
        num_players=2,
        max_size=100,
    )

    seq_len = 10
    buffers = {
        "observations": torch.zeros(seq_len, 1),
        "actions": torch.zeros(seq_len, dtype=torch.int64),
        "rewards": torch.tensor([10.0, 5.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "values": torch.zeros(seq_len),
        "policies": torch.zeros(seq_len, 2),
        "player_ids": torch.zeros(seq_len, dtype=torch.int64),
        "to_plays": torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "chances": torch.zeros(seq_len, 1),
        "episode_id": torch.ones(seq_len, dtype=torch.int64),
        "legal_masks": torch.ones(seq_len, 2, dtype=torch.bool),
        "terminated": torch.zeros(seq_len, dtype=torch.bool),
        "truncated": torch.zeros(seq_len, dtype=torch.bool),
        "done": torch.zeros(seq_len, dtype=torch.bool),
        "is_same_episode": torch.ones(seq_len, dtype=torch.bool),
        "has_valid_obs_mask": torch.ones(seq_len, dtype=torch.bool),
        "ids": torch.arange(seq_len, dtype=torch.int64),
        "training_steps": torch.zeros(seq_len, dtype=torch.int64),
    }

    indices = [0]
    batch = processor.process_batch(indices, buffers)
    values = batch["values"]
    rewards = batch["rewards"]

    # Expected Value targets (G_u):
    # G0 (at s0, p0): 10 + (-5) + 7 = 12
    # G1 (at s1, p1): 5 + (-7) = -2
    # G2 (at s2, p0): 7
    expected_values = torch.tensor([[12.0, -2.0, 7.0]])

    # Expected Reward head targets (r_u):
    # r1 (earned by p0) -> 10.0. Stored at raw_rewards[0].
    # r2 (earned by p1) -> 5.0.  Stored at raw_rewards[1].
    # r3 (earned by p0) -> 7.0.  Stored at raw_rewards[2].
    # Alignment: targets[0] matches R1 (r1).
    # expected_rewards = [0, 10, 5] (K+1=3)
    expected_rewards = torch.tensor([[0.0, 10.0, 5.0]])

    torch.testing.assert_close(
        values, expected_values, msg="Multiplayer sign flipping is incorrect for values!"
    )
    torch.testing.assert_close(
        rewards, expected_rewards, msg="Reward head targets should be mover-relative!"
    )


def test_nstep_terminal_zero_value():
    """
    Verifies that terminal states have a 0 value target and don't include
    the reward that reached them.

    Sequence:
        s0 -> r1=0 -> s1 -> r2=1 (terminal)

    Gt Targets (n=2):
        G0 = 0 + 1 + 0 = 1
        G1 = 0
    """
    unroll_steps = 1
    n_step = 2
    processor = NStepUnrollProcessor(
        unroll_steps=unroll_steps,
        n_step=n_step,
        gamma=1.0,
        num_actions=2,
        num_players=1,
        max_size=100,
    )

    seq_len = 5
    buffers = {
        "observations": torch.zeros(seq_len, 1),
        "actions": torch.zeros(seq_len, dtype=torch.int64),
        "rewards": torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
        "values": torch.zeros(seq_len),
        "policies": torch.zeros(seq_len, 2),
        "player_ids": torch.zeros(seq_len, dtype=torch.int64),
        "to_plays": torch.zeros(seq_len),
        "chances": torch.zeros(seq_len, 1),
        "episode_id": torch.ones(seq_len, dtype=torch.int64),
        "legal_masks": torch.ones(seq_len, 2, dtype=torch.bool),
        "terminated": torch.tensor([False, True, False, False, False]),
        "truncated": torch.zeros(seq_len, dtype=torch.bool),
        "done": torch.tensor([False, True, False, False, False]),
        "is_same_episode": torch.ones(seq_len, dtype=torch.bool),
        "has_valid_obs_mask": torch.ones(seq_len, dtype=torch.bool),
        "ids": torch.arange(seq_len, dtype=torch.int64),
        "training_steps": torch.zeros(seq_len, dtype=torch.int64),
    }

    # WIN ON FIRST MOVE: s0 -> r1=1.0 -> s1(terminated)
    buffers["rewards"][0] = 1.0

    indices = [0]
    batch = processor.process_batch(indices, buffers)
    values = batch["values"]

    # V0 = r1 = 1.0
    # V1 = terminal = 0.0
    expected_values = torch.tensor([[1.0, 0.0]])
    torch.testing.assert_close(
        values, expected_values, msg="Terminal value target is not zero!"
    )


def test_nstep_toplay_alignment():
    """
    Verifies that target_to_plays are correctly aligned with the sequence.
    s0 (p0) -> s1 (p1) -> s2 (p0)
    expected: [p0, p1, p2] (one-hot)
    """
    unroll_steps = 2
    n_step = 1
    processor = NStepUnrollProcessor(
        unroll_steps=unroll_steps,
        n_step=n_step,
        gamma=1.0,
        num_actions=2,
        num_players=2,
        max_size=100,
    )

    seq_len = 5
    buffers = {
        "observations": torch.zeros(seq_len, 1),
        "actions": torch.zeros(seq_len, dtype=torch.int64),
        "rewards": torch.zeros(seq_len),
        "values": torch.zeros(seq_len),
        "policies": torch.zeros(seq_len, 2),
        "to_plays": torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]),
        "chances": torch.zeros(seq_len, 1),
        "episode_id": torch.ones(seq_len, dtype=torch.int64),
        "legal_masks": torch.ones(seq_len, 2, dtype=torch.bool),
        "terminated": torch.zeros(seq_len, dtype=torch.bool),
        "truncated": torch.zeros(seq_len, dtype=torch.bool),
        "done": torch.zeros(seq_len, dtype=torch.bool),
        "ids": torch.arange(seq_len, dtype=torch.int64),
        "training_steps": torch.zeros(seq_len, dtype=torch.int64),
    }

    indices = [0]
    batch = processor.process_batch(indices, buffers)
    to_plays = batch["to_plays"]  # [B, K+1, num_players]

    # Index 0: s0 (p0) -> [1, 0]
    # Index 1: s1 (p1) -> [0, 1]
    # Index 2: s2 (p0) -> [1, 0]
    expected_tp = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
    torch.testing.assert_close(to_plays, expected_tp, msg="ToPlay targets are misaligned!")
