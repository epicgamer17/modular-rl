import pytest
import torch
import numpy as np
from replay_buffers.processors import NStepUnrollProcessor

pytestmark = pytest.mark.unit


def test_nstep_unroll_processor_is_same_game_mask():
    """
    Verifies that NStepUnrollProcessor correctly generates the is_same_game mask,
    which should track trajectory boundaries regardless of termination.
    """
    unroll_steps = 2
    n_step = 1
    num_actions = 4
    num_players = 1
    max_size = 100

    processor = NStepUnrollProcessor(
        unroll_steps=unroll_steps,
        n_step=n_step,
        gamma=0.9,
        num_actions=num_actions,
        num_players=num_players,
        max_size=max_size,
    )

    # Setup dummy buffers
    batch_size = 2
    device = torch.device("cpu")

    # observations: [max_size, C, H, W] - use small flat ones
    observations = torch.zeros((max_size, 4), device=device)
    # game_ids: [max_size]
    game_ids = torch.zeros(max_size, dtype=torch.long, device=device)
    # dones: [max_size]
    dones = torch.zeros(max_size, dtype=torch.bool, device=device)

    # Set up two trajectories
    # Trajectory 0: indices 0-4. Done at 2.
    game_ids[0:5] = 0
    dones[2] = True

    # Trajectory 1: indices 10-14. No done.
    game_ids[10:15] = 1

    buffers = {
        "observations": observations,
        "rewards": torch.zeros(max_size, device=device),
        "values": torch.zeros(max_size, device=device),
        "policies": torch.zeros((max_size, num_actions), device=device),
        "actions": torch.zeros(max_size, dtype=torch.long, device=device),
        "to_plays": torch.zeros(max_size, device=device),
        "chances": torch.zeros((max_size, 1), device=device),
        "game_ids": game_ids,
        "legal_masks": torch.ones((max_size, num_actions), device=device),
        "dones": dones,
        "terminated": dones,
        "truncated": torch.zeros(max_size, dtype=torch.bool, device=device),
        "ids": torch.arange(max_size, device=device),
        "training_steps": torch.zeros(max_size, device=device),
    }

    # Sample at index 0 (Trajectory 0) and 10 (Trajectory 1)
    indices = [0, 10]

    batch = processor.process_batch(indices, buffers)

    assert "is_same_game" in batch
    is_same_game = batch["is_same_game"]
    obs_mask = batch["obs_mask"]

    # Trajectory 0 unroll (steps 0, 1, 2)
    # game_ids are [0, 0, 0]. All same game.
    # obs_mask should be [T, T, T] (includes terminal at 2)
    # BUT if we had index 3, it would be SAME GAME but post-terminal.

    # Let's adjust Trajectory 0 to test post-terminal same game
    # Trajectory 0: index 0-2 (Game 0), index 3-4 (Game 1) - actually let's keep Game 0 but done at 1.
    buffers["game_ids"][0:5] = 0
    buffers["dones"][1] = True
    buffers["terminated"][1] = True

    batch = processor.process_batch(indices, buffers)
    is_same_game = batch["is_same_game"]
    obs_mask = batch["obs_mask"]

    # Batch 0 (idx 0):
    # k=0: idx 0, Game 0, not done before. obs_mask=T, same_game=T
    # k=1: idx 1, Game 0, done at 1. obs_mask=T (it IS the terminal state), same_game=T
    # k=2: idx 2, Game 0, was done at 1. obs_mask=F, same_game=T

    assert is_same_game[0, 0] == True
    assert is_same_game[0, 1] == True
    assert is_same_game[0, 2] == True

    assert obs_mask[0, 0] == True
    assert obs_mask[0, 1] == True
    assert obs_mask[0, 2] == False

    # This confirms is_same_game tracks the trajectory regardless of dones.
    # Now let's test a game break.
    buffers["game_ids"][12] = 99  # Break at k=2 for batch idx 10
    batch = processor.process_batch(indices, buffers)
    is_same_game = batch["is_same_game"]

    assert is_same_game[1, 0] == True  # Game 1
    assert is_same_game[1, 1] == True  # Game 1
    assert is_same_game[1, 2] == False  # Game 99
