import torch
import pytest
import numpy as np
from data.processors.nstep import NStepUnrollProcessor

pytestmark = pytest.mark.unit

def test_nstep_unroll_toplay_alignment():
    """
    Ensures that to_play targets are correctly aligned with states in the unroll.
    s0 (turn 0) -> a1 (turn 0) -> s1 (turn 1) -> a2 (turn 1) -> s2 (turn 0)
    """
    unroll_steps = 3
    n_step = 1
    gamma = 0.99
    num_actions = 4
    num_players = 2
    max_size = 100
    
    processor = NStepUnrollProcessor(
        unroll_steps=unroll_steps,
        n_step=n_step,
        gamma=gamma,
        num_actions=num_actions,
        num_players=num_players,
        max_size=max_size
    )
    
    # Create mock buffer data
    # 0 -> 1 -> 0 -> 1 -> ...
    to_plays = torch.tensor([i % 2 for i in range(max_size)], dtype=torch.int16)
    obs = torch.arange(max_size).float().unsqueeze(-1) # Each obs is just its index
    rewards = torch.arange(max_size).float() * 10 # r1=10, r2=20, ... (r_i is reward for s_{i-1}->s_i)
    dones = torch.zeros(max_size, dtype=torch.bool)
    game_ids = torch.zeros(max_size, dtype=torch.long)
    
    buffers = {
        "observations": obs,
        "to_plays": to_plays,
        "rewards": rewards,
        "dones": dones,
        "episode_ids": game_ids,
        "step_ids": torch.zeros(max_size, dtype=torch.int32),
        "values": torch.zeros(max_size),
        "policies": torch.zeros(max_size, num_actions),
        "actions": torch.zeros(max_size),
        "chances": torch.zeros(max_size, 1),
        "legal_masks": torch.ones(max_size, num_actions, dtype=torch.bool),
        "ids": torch.arange(max_size),
        "training_steps": torch.zeros(max_size),
    }
    
    # Sample starting at index 10
    indices = [10]
    batch = processor.process_batch(indices, buffers)
    
    # Check Alignment
    # s0 is at index 10. Turn should be 10 % 2 = 0.
    # s1 is at index 11. Turn should be 11 % 2 = 1.
    # s2 is at index 12. Turn should be 12 % 2 = 0.
    # s3 is at index 13. Turn should be 13 % 2 = 1.
    
    target_tp = batch["to_plays"] # [B, K+1, num_players]
    tp_players = target_tp[0].argmax(dim=-1)
    
    assert tp_players[0] == 0, f"s0 turn mismatch: expected 0, got {tp_players[0]}"
    assert tp_players[1] == 1, f"s1 turn mismatch: expected 1, got {tp_players[1]}"
    assert tp_players[2] == 0, f"s2 turn mismatch: expected 0, got {tp_players[2]}"
    assert tp_players[3] == 1, f"s3 turn mismatch: expected 1, got {tp_players[3]}"
    
    # Check Masking alignment
    tp_mask = batch["to_play_mask"][0]
    assert tp_mask[0] == False, "Root TP must be masked out"
    assert tp_mask[1] == True
    assert tp_mask[2] == True
    assert tp_mask[3] == True
    
    # Check Rewards
    # Reward contract: buffers["rewards"][i] stores the reward for transition s_i -> s_{i+1}.
    # Index 10 is s0. So r1 (for s0 -> s1) is rewards[10].
    # rewards[10] = 10 * 10 = 100.
    target_rewards = batch["rewards"][0]
    assert target_rewards[1] == 100.0, f"r1 mismatch: expected 100, got {target_rewards[1]}. Logic: if s0 is at index 10, r1 must be rewards[10]."
    assert target_rewards[2] == 110.0, f"r2 mismatch: expected 110, got {target_rewards[2]}"
    assert target_rewards[0] == 0.0, "r0 must be 0 padding"

def test_nstep_unroll_terminal_masking():
    """
    Ensures that unrolls past a terminal state are correctly masked.
    s0 -> s1 (TERM) -> [s2 invalid]
    """
    unroll_steps = 3
    processor = NStepUnrollProcessor(
        unroll_steps=unroll_steps, n_step=1, gamma=1.0, 
        num_actions=2, num_players=2, max_size=100
    )
    
    to_plays = torch.tensor([i % 2 for i in range(100)], dtype=torch.int16)
    obs = torch.arange(100).float().unsqueeze(-1)
    
    # TERMINAL LOGIC:
    # raw_dones[i] is True if transition s_i -> s_{i+1} was terminal.
    # If dones[11] is True, it means s11 was terminal state reached from s10.
    # Sample index 10 (s0).
    # s0 (10) ok. s1 (11) is terminal. s2 (12) is invalid.
    dones = torch.zeros(100, dtype=torch.bool)
    dones[11] = True 
    
    buffers = {
        "observations": obs,
        "to_plays": to_plays,
        "rewards": torch.ones(100),
        "dones": dones,
        "terminated": dones,
        "episode_ids": torch.zeros(100, dtype=torch.long),
        "step_ids": torch.zeros(100, dtype=torch.int32),
        "values": torch.zeros(100),
        "policies": torch.zeros(100, 2),
        "actions": torch.zeros(100),
        "chances": torch.zeros(100, 1),
        "legal_masks": torch.ones(100, 2, dtype=torch.bool),
        "ids": torch.arange(100),
        "training_steps": torch.zeros(100),
    }
    
    batch = processor.process_batch([10], buffers)
    
    tp_mask = batch["to_play_mask"][0]
    reward_mask = batch["reward_mask"][0]
    
    # Step 1: s1 exists (terminal)
    assert tp_mask[1] == True, "s1 (terminal) should be included in loss"
    assert reward_mask[1] == True, "r1 (reward reaching terminal) should be included"
    
    # Step 2: s2 does not exist
    assert tp_mask[2] == False, f"s2 (post-terminal) must be masked out. Mask: {tp_mask}"
    assert reward_mask[2] == False, "r2 (post-terminal transition) must be masked out"

def test_nstep_unroll_same_game_masking():
    """
    Ensures that unrolls crossing game boundaries are masked even if not terminal.
    Game 0 (ends at 11) | Game 1 (starts at 12)
    """
    unroll_steps = 3
    processor = NStepUnrollProcessor(
        unroll_steps=unroll_steps, n_step=1, gamma=1.0, 
        num_actions=2, num_players=2, max_size=100
    )
    
    game_ids = torch.zeros(100, dtype=torch.long)
    game_ids[10] = 7
    game_ids[11] = 7
    game_ids[12] = 8 # Global change at index 12
    
    buffers = {
        "observations": torch.arange(100).float().unsqueeze(-1),
        "to_plays": torch.zeros(100, dtype=torch.int16),
        "rewards": torch.ones(100),
        "dones": torch.zeros(100, dtype=torch.bool),
        "episode_ids": game_ids,
        "step_ids": torch.zeros(100, dtype=torch.int32),
        "values": torch.zeros(100),
        "policies": torch.zeros(100, 2),
        "actions": torch.zeros(100),
        "chances": torch.zeros(100, 1),
        "legal_masks": torch.ones(100, 2, dtype=torch.bool),
        "ids": torch.arange(100),
        "training_steps": torch.zeros(100),
    }
    
    batch = processor.process_batch([10], buffers)
    
    tp_mask = batch["to_play_mask"][0]
    is_same_episode = batch["is_same_episode"][0]
    
    # Debug info
    print(f"DEBUG: same_game: {is_same_episode}")
    print(f"DEBUG: tp_mask: {tp_mask}")

    assert is_same_episode[0] == True
    assert is_same_episode[1] == True
    assert is_same_episode[2] == False, f"Index 2 (buffer 12) is game 8, but base was 7. same_game: {is_same_episode}"
    
    assert tp_mask[1] == True # s1 is same game
    assert tp_mask[2] == False, f"s2 (different game) must be masked out. Mask: {tp_mask}"
