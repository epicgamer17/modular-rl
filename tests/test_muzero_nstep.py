import torch
import numpy as np
import pytest
from replay_buffers.processors import MuZeroUnrollOutputProcessor


def test_nstep_value_single_player():
    """
    Test n-step value calculation for single player (no sign flips).
    """
    unroll_steps = 2
    n_step = 2
    gamma = 1.0
    num_actions = 2
    num_players = 1

    proc = MuZeroUnrollOutputProcessor(
        unroll_steps=unroll_steps,
        n_step=n_step,
        gamma=gamma,
        num_actions=num_actions,
        num_players=num_players,
        max_size=10,
    )

    # Setup dummy buffer
    # Observations: [0, 1, 2, 3, 4]
    # Rewards: [10, 20, 30, 40]
    # Values: [0, 0, 0, 0, 100]
    # To Play: [0, 0, 0, 0, 0]

    batch_size = 1
    raw_rewards = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    raw_values = torch.tensor([[0.0, 0.0, 0.0, 0.0, 100.0]])
    raw_to_plays = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.int16)
    raw_dones = torch.tensor([[False, False, False, False, False]])
    valid_mask = torch.tensor([[True, True, True, True, True]])

    # Internal method test
    target_values, target_rewards = proc._compute_n_step_targets(
        batch_size=batch_size,
        raw_rewards=raw_rewards,
        raw_values=raw_values,
        raw_to_plays=raw_to_plays,
        raw_dones=raw_dones,
        valid_mask=valid_mask,
        device="cpu",
    )

    # k=0 (root S0): R1 + R2 + V2 = 10 + 20 + 0 = 30
    # k=1 (S1): R2 + R3 + V3 = 20 + 30 + 0 = 50
    # k=2 (S2): R3 + R4 + V4 = 30 + 40 + 100 = 170

    assert target_values[0, 0] == 30.0
    assert target_values[0, 1] == 50.0
    assert target_values[0, 2] == 170.0

    # target_rewards: root=0, k=1 -> R1=10, k=2 -> R2=20
    assert target_rewards[0, 0] == 0.0
    assert target_rewards[0, 1] == 10.0
    assert target_rewards[0, 2] == 20.0


def test_nstep_value_multi_player():
    """
    Test n-step value calculation with player sign flips.
    """
    unroll_steps = 1
    n_step = 2
    gamma = 1.0
    num_actions = 2
    num_players = 2

    proc = MuZeroUnrollOutputProcessor(
        unroll_steps=unroll_steps,
        n_step=n_step,
        gamma=gamma,
        num_actions=num_actions,
        num_players=num_players,
        max_size=10,
    )

    # Players: [0, 1, 0, 1]
    # Rewards: [10, 20, 30]
    # Values: [0, 0, 0, 100]
    raw_to_plays = torch.tensor([[0, 1, 0, 1]], dtype=torch.int16)
    raw_rewards = torch.tensor([[10.0, 20.0, 30.0]])
    raw_values = torch.tensor([[0.0, 0.0, 0.0, 100.0]])
    raw_dones = torch.tensor([[False, False, False, False]])
    valid_mask = torch.tensor([[True, True, True, True]])

    target_values, _ = proc._compute_n_step_targets(
        batch_size=1,
        raw_rewards=raw_rewards,
        raw_values=raw_values,
        raw_to_plays=raw_to_plays,
        raw_dones=raw_dones,
        valid_mask=valid_mask,
        device="cpu",
    )

    # k=0 (root S0, player 0):
    # sign(R1, p=0) * R1 + sign(R2, p=1) * R2 + sign(V2, p=0) * V2
    # Wait, sign logic in processor:
    # sign = torch.where(current_player == step_player, 1.0, -1.0)
    # k=0, current_player=0
    # step 0: player 0, sign=+1. R1=10
    # step 1: player 1, sign=-1. R2=20
    # V bootstrap (k=n_step):
    # v_sign = torch.where(current_player == boot_player, 1.0, -1.0)
    # step 2: player 0, sign=+1. V2=0
    # Total = 10 - 20 + 0 = -10

    assert target_values[0, 0] == -10.0


def test_value_prefix_logic():
    """
    Test value prefix (EfficientZero style accumulation).
    """
    proc = MuZeroUnrollOutputProcessor(
        unroll_steps=3,
        n_step=1,
        gamma=1.0,
        num_actions=2,
        num_players=1,
        max_size=10,
        value_prefix=True,
        lstm_horizon_len=2,
    )

    raw_rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])
    raw_to_plays = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.int16)
    valid_mask = torch.tensor([[True, True, True, True, True, True]])

    _, target_rewards = proc._compute_n_step_targets(
        batch_size=1,
        raw_rewards=raw_rewards,
        raw_values=torch.zeros((1, 6)),
        raw_to_plays=raw_to_plays,
        raw_dones=torch.zeros((1, 6), dtype=torch.bool),
        valid_mask=valid_mask,
        device="cpu",
    )

    # Result: [0, 1, 2, 1]

    assert target_rewards[0, :4].tolist() == [0.0, 1.0, 2.0, 1.0]


def test_nstep_specific_user_case():
    """
    Test n-step values for the specific example provided by the user.
    values = [0.2, -0.1, 0.4, 0.5]
    rewards = [1.0, -2.0, 3.0]
    actions = [0, 1, 2]
    policies = [
        torch.tensor([0.7, 0.3]),
        torch.tensor([0.2, 0.8]),
        torch.tensor([0.5, 0.5]),
        torch.tensor([0.5, 0.5]),
    ]
    infos = [
        {"player": 0},
        {"player": 1},
        {"player": 0},
        {"player": 1},
    ]
    index = 0
    num_unroll_steps = 2
    n_step = 2
    gamma = 0.99
    """
    proc = MuZeroUnrollOutputProcessor(
        unroll_steps=2, n_step=2, gamma=0.99, num_actions=2, num_players=2, max_size=10
    )

    raw_rewards = torch.tensor([[1.0, -2.0, 3.0]])
    raw_values = torch.tensor([[0.2, -0.1, 0.4, 0.5]])
    raw_to_plays = torch.tensor([[0, 1, 0, 1]], dtype=torch.int16)
    raw_dones = torch.tensor([[False, False, False, False]])
    valid_mask = torch.tensor([[True, True, True, True]])

    target_values, target_rewards = proc._compute_n_step_targets(
        batch_size=1,
        raw_rewards=raw_rewards,
        raw_values=raw_values,
        raw_to_plays=raw_to_plays,
        raw_dones=raw_dones,
        valid_mask=valid_mask,
        device="cpu",
    )

    # Computation for index=0, u=0 (root):
    # n_step=2
    # k=0: r_idx=0, player=0, sign=+1. R=1.0. val += 1.0
    # k=1: r_idx=1, player=1, sign=-1. R=-2.0. val += 0.99 * (-1 * -2.0) = 1.98
    # bootstrap k=2: boot_idx=2, player=0, sign=+1. V=0.4. val += (0.99^2) * (+1 * 0.4) = 0.9801 * 0.4 = 0.39204
    # Total = 1.0 + 1.98 + 0.39204 = 3.37204

    assert pytest.approx(float(target_values[0, 0]), rel=1e-5) == 3.37204

    # Computation for index=0, u=1:
    # k=0: r_idx=1, current_player=1, acting_player=1, sign=+1. R=-2.0. val += -2.0
    # k=1: r_idx=2, current_player=1, acting_player=0, sign=-1. R=3.0. val += 0.99 * (-1 * 3.0) = -2.97
    # bootstrap k=2: boot_idx=3, player_at_3=1, sign=+1. V=0.5. val += (0.99^2) * (+1 * 0.5) = 0.9801 * 0.5 = 0.49005
    # Total = -2.0 - 2.97 + 0.49005 = -4.47995

    assert pytest.approx(float(target_values[0, 1]), rel=1e-5) == -4.47995
