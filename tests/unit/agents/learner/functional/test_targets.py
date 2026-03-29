import math
import torch
import pytest
from agents.learner.functional.returns import compute_unrolled_n_step_targets

pytestmark = pytest.mark.unit


def test_muzero_unrolled_n_step_targets_math():
    """
    Verify the N-step return calculation for MuZero unrolled sequences.

    Formula: G_t = sum_{i=0}^{N-1} gamma^i r_{t+1+i} + gamma^N V_{t+N}

    Setup:
    - Trajectory Length: 10 steps (indices 0-9)
    - Rewards (r1, r2, r3, r4, r5, ...): [1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
    - Values (V(s_t)): [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    - N-step (N): 5
    - Unroll length (K): 3 (targets needed for t=0, 1, 2, 3)
    - Discount factor (gamma): 0.9

    Hand-calculated Expected Targets:
    - t=0: r1 + gamma r2 + gamma^2 r3 + gamma^3 r4 + gamma^4 r5 + gamma^5 V5
           = 1 + 0 + 0.81(1) + 0 + 0.6561(1) + 0.59049(0.5)
           = 1 + 0.81 + 0.6561 + 0.295245 = 2.761345
    - t=1: r2 + gamma r3 + gamma^2 r4 + gamma^3 r5 + gamma^4 r6 + gamma^5 V6
           = 0 + 0.9(1) + 0 + 0.729(1) + 0 + 0.59049(0.5)
           = 0.9 + 0.729 + 0.295245 = 1.924245
    - t=2: r3 + gamma r4 + gamma^2 r5 + gamma^3 r6 + gamma^4 r7 + gamma^5 V7
           = 1 + 0 + 0.81(1) + 0 + 0 + 0.59049(0.5)
           = 1.81 + 0.295245 = 2.105245
    - t=3: r4 + gamma r5 + gamma^2 r6 + gamma^3 r7 + gamma^4 r8 + gamma^5 V8
           = 0 + 0.9(1) + 0 + 0 + 0 + 0.59049(0.5)
           = 0.9 + 0.295245 = 1.195245
    """
    # 1. Setup Data
    B = 1
    L = 10
    device = torch.device("cpu")

    raw_rewards = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device
    )
    raw_values = torch.tensor(
        [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], device=device
    )
    raw_to_plays = torch.zeros((B, L), dtype=torch.long, device=device)
    raw_terminated = torch.zeros((B, L), dtype=torch.bool, device=device)
    valid_mask = torch.ones((B, L), dtype=torch.bool, device=device)

    gamma = 0.9
    n_step = 5
    unroll_steps = 3  # K=3

    # 2. Compute Targets
    target_values, target_rewards = compute_unrolled_n_step_targets(
        raw_rewards=raw_rewards,
        raw_values=raw_values,
        raw_to_plays=raw_to_plays,
        raw_terminated=raw_terminated,
        valid_mask=valid_mask,
        gamma=gamma,
        n_step=n_step,
        unroll_steps=unroll_steps,
    )

    # Expected results for value targets t=0, 1, 2, 3
    expected_values = torch.tensor(
        [2.761345, 1.924245, 2.105245, 1.195245], device=device
    )

    # Assert Value targets
    torch.testing.assert_close(target_values[0], expected_values, atol=1e-6, rtol=1e-6)

    # Expected rewards for transitions (instant rewards)
    # Target rewards are state-aligned: Root (t=0) has 0 reward.
    # t=1 target reward is reachability from t=0 -> t=1 (r1).
    expected_rewards = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device)

    # Assert Rewards
    torch.testing.assert_close(
        target_rewards[0], expected_rewards, atol=1e-6, rtol=1e-6
    )


def test_muzero_targets_terminal_state_invariant():
    """
    Ensure the value target of a terminal state is 0.0 and discounting stops.

    Setup:
    - Terminated at index 4 (t=5).
    - t=0 calculation: N=5 should reach index 4 and capture r1..r5, but NO bootstrap.
    """
    B, L = 1, 10
    device = torch.device("cpu")

    raw_rewards = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device
    )
    raw_values = torch.tensor(
        [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], device=device
    )
    raw_terminated = torch.zeros((B, L), dtype=torch.bool, device=device)
    raw_terminated[0, 4] = True  # Terminate on r5 transition

    valid_mask = torch.ones((B, L), dtype=torch.bool, device=device)

    gamma = 0.9
    n_step = 5
    unroll_steps = 1  # Root and 1 transition

    # Compute Targets
    target_values, _ = compute_unrolled_n_step_targets(
        raw_rewards=raw_rewards,
        raw_values=raw_values,
        raw_to_plays=torch.zeros((B, L), dtype=torch.long),
        raw_terminated=raw_terminated,
        valid_mask=valid_mask,
        gamma=gamma,
        n_step=n_step,
        unroll_steps=unroll_steps,
    )

    # t=0: r1 + gamma r2 + gamma^2 r3 + gamma^3 r4 + gamma^4 r5 + (0 because terminated)
    # 1 + 0 + 0.81 + 0 + 0.6561 = 2.4661
    expected_t0 = 2.4661

    # t=1: r2 + gamma r3 + gamma^2 r4 + gamma^3 r5 + (0 because terminated)
    # 0 + 0.9(1) + 0 + 0.729(1) = 1.629
    expected_t1 = 1.629

    assert math.isclose(target_values[0, 0].item(), expected_t0, rel_tol=1e-6)
    assert math.isclose(target_values[0, 1].item(), expected_t1, rel_tol=1e-6)


def test_value_prefix_lstm_horizon_reset():
    """
    CONTRACT: When horizon_id hits a multiple of lstm_horizon_len,
    the accumulated value_prefix must reset to 0.
    """
    B, L = 1, 10
    device = torch.device("cpu")

    raw_rewards = torch.ones((B, L), device=device)
    raw_values = torch.zeros((B, L), device=device)
    raw_to_plays = torch.zeros((B, L), dtype=torch.long, device=device)
    raw_terminated = torch.zeros((B, L), dtype=torch.bool, device=device)
    valid_mask = torch.ones((B, L), dtype=torch.bool, device=device)

    # Run the system under test
    _, target_rewards = compute_unrolled_n_step_targets(
        raw_rewards=raw_rewards,
        raw_values=raw_values,
        raw_to_plays=raw_to_plays,
        raw_terminated=raw_terminated,
        valid_mask=valid_mask,
        gamma=1.0,
        unroll_steps=5,
        n_step=3,
        lstm_horizon_len=2,  # Reset every 2 steps
        value_prefix=True,
    )

    # Analytical expected accumulated prefix
    # s0: 0
    # s1: r0 = 1
    # s2: r1 = 1 (reset at t=2 starts from r1)
    # s3: r1+r2 = 2
    # s4: r3 = 1 (reset at t=4 starts from r3)
    # s5: r3+r4 = 2
    expected_rewards = torch.tensor(
        [[0.0, 1.0, 1.0, 2.0, 1.0, 2.0]], dtype=torch.float32, device=device
    )
    torch.testing.assert_close(target_rewards, expected_rewards)


def test_two_player_n_step_sign_flips():
    """
    MATH: Two-player alternating value_prefix must flip the sign
    for rewards from the opposing player.
    """
    B, L = 1, 10
    device = torch.device("cpu")

    # rewards = [1, 2, 3, 4, ...]
    raw_rewards = torch.arange(1, L + 1, dtype=torch.float32, device=device).unsqueeze(
        0
    )
    raw_values = torch.zeros((B, L), device=device)

    # Alternating players: 0, 1, 0, 1...
    raw_to_plays = (torch.arange(L, device=device) % 2).unsqueeze(0)
    raw_terminated = torch.zeros((B, L), dtype=torch.bool, device=device)
    valid_mask = torch.ones((B, L), dtype=torch.bool, device=device)

    _, target_rewards = compute_unrolled_n_step_targets(
        raw_rewards=raw_rewards,
        raw_values=raw_values,
        raw_to_plays=raw_to_plays,
        raw_terminated=raw_terminated,
        valid_mask=valid_mask,
        gamma=1.0,
        unroll_steps=3,
        n_step=3,
        value_prefix=True,
        lstm_horizon_len=None,  # Large horizon implicitly
    )

    # u=0 -> 0
    # u=1 -> P0 gets +1 -> prefix = 1
    # u=2 -> P1 gets +2 -> prefix = 1 + (-2) = -1
    # u=3 -> P0 gets +3 -> prefix = -1 + 3 = 2
    expected_rewards = torch.tensor(
        [[0.0, 1.0, -1.0, 2.0]], dtype=torch.float32, device=device
    )
    torch.testing.assert_close(target_rewards, expected_rewards)


def test_discounted_value_prefix():
    """
    MATH: Value prefix targets are raw cumulative sums and DO NOT use gamma
    (EfficientZero cumulative reward head predicts the sum).
    """
    from agents.learner.functional.returns import compute_unrolled_n_step_targets

    B, L = 1, 10
    device = torch.device("cpu")

    raw_rewards = torch.ones((B, L), device=device)
    raw_values = torch.zeros((B, L), device=device)
    raw_to_plays = torch.zeros((B, L), dtype=torch.long, device=device)
    raw_terminated = torch.zeros((B, L), dtype=torch.bool, device=device)
    valid_mask = torch.ones((B, L), dtype=torch.bool, device=device)

    # Even with gamma=0.9, the prefix should be 1.0, 2.0, 3.0...
    _, target_rewards = compute_unrolled_n_step_targets(
        raw_rewards=raw_rewards,
        raw_values=raw_values,
        raw_to_plays=raw_to_plays,
        raw_terminated=raw_terminated,
        valid_mask=valid_mask,
        gamma=0.9,
        unroll_steps=3,
        n_step=3,
        value_prefix=True,
        lstm_horizon_len=None,
    )

    expected_rewards = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32, device=device
    )
    torch.testing.assert_close(target_rewards, expected_rewards)
