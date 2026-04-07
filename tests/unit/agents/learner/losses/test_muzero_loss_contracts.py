import torch
import pytest
import numpy as np
from learner.pipeline.target_builders import (
    SequenceMaskBuilder,
    SequencePadder,
    TargetBuilderPipeline,
)

# Module-level marker for unit tests
pytestmark = pytest.mark.unit


def test_muzero_loss_masking_contracts():
    """
    Verifies MuZero loss masking contracts:
    1. to_play: root masked, terminal valid, post-terminal masked.
    2. reward: root masked, terminal valid, post-terminal masked.
    3. policy: terminal masked, post-terminal masked, pre-terminal valid.
    4. value: valid everywhere in same game.
    """
    builder = SequenceMaskBuilder()

    # B=1, T=5. Terminal at t=3.
    # index 0: root
    # index 1: step 1
    # index 2: step 2
    # index 3: terminal state
    # index 4: post-terminal (outside same game)

    B, T = 1, 5
    terminal_idx = 3

    current_targets = {
        "actions": torch.zeros((B, T, 1)),  # [B, T, ...]
    }

    # is_same_game: True for t=0..3, False for t=4
    is_same_game = torch.zeros((B, T), dtype=torch.bool)
    is_same_game[:, : terminal_idx + 1] = True

    # has_valid_obs_mask: same as is_same_game for this test
    has_valid_obs_mask = is_same_game.clone()

    # dones: True only at terminal_idx
    dones = torch.zeros((B, T), dtype=torch.bool)
    dones[:, terminal_idx] = True

    batch = {
        "is_same_game": is_same_game,
        "has_valid_obs_mask": has_valid_obs_mask,
        "dones": dones,
    }

    # Run builder
    builder.build_targets(batch, {}, None, current_targets)

    # --- Assert to_play_mask ---
    tp_mask = current_targets["to_play_mask"]
    assert tp_mask[0, 0] == False, "to_play should be masked on root (t=0)"
    assert tp_mask[0, 1] == True, "to_play should be valid on intermediate step"
    assert (
        tp_mask[0, terminal_idx] == True
    ), "to_play should be valid on terminal state (t=3)"
    assert (
        tp_mask[0, terminal_idx + 1] == False
    ), "to_play should be masked post-terminal (t=4)"

    # --- Assert reward_mask ---
    # User requirement: reward should be valid post-terminal to train self-absorbing states to 0
    rew_mask = current_targets["reward_mask"]
    assert rew_mask[0, 0] == False, "reward should be masked on root (t=0)"
    assert rew_mask[0, 1] == True, "reward should be valid on intermediate step"
    assert (
        rew_mask[0, terminal_idx] == True
    ), "reward should be valid on terminal state (prediction of reward entering s_3)"
    assert (
        rew_mask[0, terminal_idx + 1] == False
    ), "reward should be valid post-terminal (t=4) to learn 0"

    # --- Assert policy_mask ---
    pol_mask = current_targets["policy_mask"]
    assert pol_mask[0, 0] == True, "policy should be valid on root"
    assert pol_mask[0, 1] == True, "policy should be valid on intermediate step"
    # CRITICAL: Policy must be masked on terminal states
    assert (
        pol_mask[0, terminal_idx] == False
    ), f"policy should be masked on terminal state (t={terminal_idx})"
    assert (
        pol_mask[0, terminal_idx + 1] == False
    ), "policy should be masked post-terminal"

    # --- Assert value_mask ---
    # User requirement: value should be valid post-terminal
    val_mask = current_targets["value_mask"]
    assert val_mask[0, 0] == True, "value should be valid on root"
    assert val_mask[0, terminal_idx] == True, "value should be valid on terminal state"
    assert val_mask[0, terminal_idx + 1] == False, "value should be valid post-terminal"


def test_muzero_padding_contracts():
    """
    Verifies that SequencePadder correctly handles t=0 padding for transition-aligned data.
    """
    K = 4
    T = K + 1
    B = 1

    padder = SequencePadder(unroll_steps=K)

    # Transition data (rewards, actions) are length K
    rewards = torch.ones((B, K))

    current_targets = {
        "rewards": rewards,
        "actions": torch.zeros((B, T, 1)),  # Anchor to determine T
    }

    padder.build_targets({}, {}, None, current_targets)

    assert current_targets["rewards"].shape == (B, T)
    assert (
        current_targets["rewards"][0, 0] == 0.0
    ), "Transition data at t=0 should be zero-padded"


def test_muzero_value_terminal_invariant():
    """
    Verifies terminal value invariant (V=0) and padding post-terminal.
    This logic lives in NStepUnrollProcessor.
    """
    from data.processors import NStepUnrollProcessor

    UNROLL_STEPS = 4
    N_STEP = 3
    GAMMA = 0.99
    NUM_ACTIONS = 2
    NUM_PLAYERS = 2
    MAX_SIZE = 100

    processor = NStepUnrollProcessor(
        unroll_steps=UNROLL_STEPS,
        n_step=N_STEP,
        gamma=GAMMA,
        num_actions=NUM_ACTIONS,
        num_players=NUM_PLAYERS,
        max_size=MAX_SIZE,
    )

    # Create a synthetic buffer
    device = torch.device("cpu")
    buffers = {
        "observations": torch.zeros((MAX_SIZE, 1)),
        "rewards": torch.ones((MAX_SIZE,)),
        "values": torch.ones((MAX_SIZE,)),
        "policies": torch.zeros((MAX_SIZE, NUM_ACTIONS)),
        "actions": torch.zeros((MAX_SIZE,), dtype=torch.long),
        "to_plays": torch.zeros((MAX_SIZE,), dtype=torch.long),
        "chances": torch.zeros((MAX_SIZE, 1), dtype=torch.long),
        "game_ids": torch.zeros((MAX_SIZE,), dtype=torch.long),
        "legal_masks": torch.ones((MAX_SIZE, NUM_ACTIONS), dtype=torch.bool),
        "terminated": torch.zeros((MAX_SIZE,), dtype=torch.bool),
        "truncated": torch.zeros((MAX_SIZE,), dtype=torch.bool),
        "training_steps": torch.zeros((MAX_SIZE,), dtype=torch.long),
        "ids": torch.arange(MAX_SIZE),
    }

    # Terminal at index 2 (s2 is terminal)
    # Sequence starting at index 0: s0, s1, s2(terminal)
    buffers["terminated"][2] = True

    indices = [0]
    batch = processor.process_batch(indices, buffers)

    target_values = batch["values"]  # [B, T]
    target_rewards = batch["rewards"]  # [B, T]
    obs_valid_mask = batch["has_valid_obs_mask"]  # [B, T]

    # t=0: s0
    # t=1: s1
    # t=2: s2 (terminal)
    # t=3: s2 (padded/post-terminal)

    assert obs_valid_mask[0, 0] == True
    assert obs_valid_mask[0, 2] == True
    assert obs_valid_mask[0, 3] == False

    # Value at terminal state s2 (index 2) should be 0
    assert (
        target_values[0, 2] == 0.0
    ), f"Terminal value should be 0.0, got {target_values[0, 2]}"

    # Value post-terminal should be 0
    assert target_values[0, 3] == 0.0

    # Reward at root should be 0
    assert target_rewards[0, 0] == 0.0
    # Reward entering terminal s2 should be valid (1.0)
    assert target_rewards[0, 2] == 1.0
    # Reward post-terminal should be 0
    assert target_rewards[0, 3] == 0.0
