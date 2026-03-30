import pytest
import torch
from agents.learner.target_builders import SequenceMaskBuilder

pytestmark = pytest.mark.unit

def test_sequence_mask_builder_contract():
    """
    Verifies that SequenceMaskBuilder generates the correct masks for MuZero losses.
    Contract:
    - value_mask: covers s0..sk (includes root and terminal).
    - reward_mask: covers s1..sk (masks out root, includes terminal? wait, check this).
    - policy_mask: covers s0..s_{k-1} (includes root, masks out terminal).
    - to_play_mask: covers s1..sk (masks out root, includes terminal).
    """
    B, T = 1, 6
    unroll_steps = 5
    device = torch.device("cpu")
    
    # Mock batch
    # Sequence: s0, s1, s2, s3(Terminal), s4(Post), s5(Post)
    # k = 3
    is_same_game = torch.tensor([[True, True, True, True, True, True]], device=device)
    # dones [B, T]. Root is False (s0), s1 False, s2 False, s3 True, s4 True(sticky), s5 True
    dones = torch.tensor([[False, False, False, True, True, True]], device=device)
    
    # has_valid_action_mask: same_game & ~post_done & ~raw_dones
    # post_done: [F, F, F, F, T, T] (done at s3, so s4+ is post_done)
    post_done = torch.tensor([[False, False, False, False, True, True]], device=device)
    # raw_dones: [F, F, F, T, T, T]
    # has_valid_action_mask: [T, T, T, F, F, F] (s0, s1, s2 are valid transitions)
    has_valid_action_mask = is_same_game & (~post_done) & (~dones)
    
    batch = {
        "is_same_game": is_same_game,
        "has_valid_action_mask": has_valid_action_mask,
        "dones": dones,
        "actions": torch.zeros((B, unroll_steps), device=device) # T-1 actions is enough for T states? 
                                                                # Actually builder expects actions [B, T] because they were padded.
    }
    # Fix actions shape to match T states if padded
    batch["actions"] = torch.zeros((B, T), device=device)
    
    current_targets = {
        "actions": batch["actions"]
    }
    
    builder = SequenceMaskBuilder()
    builder.build_targets(batch, {}, None, current_targets)
    
    value_mask = current_targets["value_mask"]
    reward_mask = current_targets["reward_mask"]
    policy_mask = current_targets["policy_mask"]
    to_play_mask = current_targets["to_play_mask"]
    
    # 1. Value Mask: Includes everything up to terminal (and depending on base_mask, post-terminal)
    # Our base_mask is all True, so value_mask should be all True.
    assert value_mask[0].tolist() == [True, True, True, True, True, True]
    
    # 2. Reward Mask: Root False, rest (including post-terminal) True IF same_game.
    # index 0 (root) False. s1..s5 True.
    assert reward_mask[0].tolist() == [False, True, True, True, True, True]
    
    # 3. Policy Mask: Includes root, masks out terminal and post-terminal
    # s0, s1, s2 are valid. s3 (terminal) is False. s4, s5 are False.
    assert policy_mask[0].tolist() == [True, True, True, False, False, False]
    
    # 4. To-Play Mask: Masks out root, includes terminal, masks out post-terminal
    # s0 False, s1 True, s2 True, s3(Terminal) True, s4, s5 False.
    expected_to_play = [False, True, True, True, False, False]
    
    actual_to_play = to_play_mask[0].tolist()
    print(f"Actual To-Play Mask: {actual_to_play}")
    assert actual_to_play == expected_to_play, f"To-Play mask mismatch. Expected {expected_to_play}, got {actual_to_play}"
    
    # 5. Consistency Mask: Same as to_play (valid unrolled dynamics checkpoints)
    consistency_mask = current_targets["consistency_mask"]
    assert consistency_mask[0].tolist() == expected_to_play

if __name__ == "__main__":
    test_sequence_mask_builder_contract()
