import pytest
import torch
import numpy as np
from replay_buffers.processors import NStepUnrollProcessor

pytestmark = pytest.mark.unit


def get_mock_buffers(L=20):
    # Buffer elements should be flat arrays [L, ...] in the replay buffer
    # so they can be indexed by all_indices [B, window]
    raw_rewards = torch.zeros(L)
    raw_rewards[:8] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    raw_values = torch.zeros(L)
    raw_to_plays = torch.zeros(L)
    raw_game_ids = torch.zeros(L)
    raw_policies = torch.zeros(L, 4)
    raw_actions = torch.zeros(L)
    raw_chances = torch.zeros(L, 1)  # [L, 1] as per processors.py line 534
    raw_terminated = torch.zeros(L, dtype=torch.bool)
    raw_truncated = torch.zeros(L, dtype=torch.bool)
    raw_legal_masks = torch.ones(L, 4, dtype=torch.bool)
    raw_ids = torch.zeros(L)
    raw_training_steps = torch.zeros(L)

    return {
        "observations": torch.randn(L, 4),
        "rewards": raw_rewards,
        "values": raw_values,
        "to_plays": raw_to_plays,
        "chances": raw_chances,
        "game_ids": raw_game_ids,
        "policies": raw_policies,
        "actions": raw_actions,
        "terminated": raw_terminated,
        "truncated": raw_truncated,
        "legal_masks": raw_legal_masks,
        "ids": raw_ids,
        "training_steps": raw_training_steps,
    }


def test_nstep_unroll_processor_value_prefix_accumulation():
    """
    Verifies that NStepUnrollProcessor correctly calculates target_rewards
    as cumulative sums when value_prefix=True, and resets according to lstm_horizon_len.
    """
    processor = NStepUnrollProcessor(
        unroll_steps=5,
        n_step=3,
        gamma=0.9,
        num_actions=4,
        num_players=2,
        max_size=100,
        lstm_horizon_len=3,
        value_prefix=True,
    )

    buffers = get_mock_buffers()
    indices = [0]
    result = processor.process_batch(indices, buffers)

    # target_rewards[u] should be the prefix for state u.
    # horizon = 3
    # u=0: prefix=0
    # u=1: prefix=r[0] = 1.0
    # u=2: prefix=r[0]+r[1] = 1.0+2.0 = 3.0
    # u=3: prefix=0 (reset at index 3, if horizon_id % 3 == 0)
    #   Wait, processor logic (re-calculating based on code):
    #   u=1: h_id=0 % 3 == 0 -> prefix=0. h_id=1. prefix += r[0](1). target[1]=1.
    #   u=2: h_id=1. h_id=2. prefix += r[1](2). target[2]=3.
    #   u=3: h_id=2. h_id=3. prefix += r[2](3). target[3]=6.
    #   u=4: h_id=3 % 3 == 0 -> prefix=0. h_id=4. prefix += r[3](4). target[4]=4.
    #   u=5: h_id=4. h_id=5. prefix += r[4](5). target[5]=9.

    expected_rewards = torch.tensor([0.0, 1.0, 3.0, 6.0, 4.0, 9.0])
    actual_rewards = result["rewards"][0, :6]

    assert torch.allclose(actual_rewards, expected_rewards)


def test_nstep_unroll_processor_no_value_prefix():
    """Verifies that with value_prefix=False, target_rewards are just shifted raw_rewards."""
    processor = NStepUnrollProcessor(
        unroll_steps=5,
        n_step=3,
        gamma=0.9,
        num_actions=4,
        num_players=2,
        max_size=100,
        value_prefix=False,
    )

    buffers = get_mock_buffers()
    indices = [0]
    result = processor.process_batch(indices, buffers)

    # target_rewards[u] should be raw_rewards[u-1] for u > 0, and 0 for u=0
    expected_rewards = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    actual_rewards = result["rewards"][0, :6]

    assert torch.allclose(actual_rewards, expected_rewards)
