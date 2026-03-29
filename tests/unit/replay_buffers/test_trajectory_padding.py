import pytest
import torch
import numpy as np

from replay_buffers.processors import NStepUnrollProcessor
from replay_buffers.modular_buffer import ModularReplayBuffer, BufferConfig

pytestmark = pytest.mark.unit

def test_trajectory_padding_contract():
    """
    Tier 1 Unit Test: Trajectory Padding Contract
    - Mocks an episode that terminates at step 3.
    - Unrolls for K=5 steps.
    - Asserts the sequence shape corresponds to the unroll window.
    - Asserts that target indices past the termination explicitly handle padding 
      (e.g., target policies = 0.0, to_plays = 0.0).
    - Asserts the generated loss mask explicitly ignores steps after termination.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    
    max_size = 10
    num_actions = 4
    num_players = 1
    unroll_steps = 5
    n_step = 1
    gamma = 0.99
    
    # Let's say step 3 means index 3 is a done (0-indexed: 0, 1, 2, 3 -> step 3 is terminal).
    # Wait, "step 3" could mean sequence length 3. Let's make index 2 terminal.
    # Indices: 
    # 0: not done
    # 1: not done
    # 2: done
    # 3: not done (next episode starts)
    # 4: not done
    # 5: not done
    # 6: not done
    
    configs = [
        BufferConfig("observations", shape=(1,), dtype=torch.float32),
        BufferConfig("actions", shape=(), dtype=torch.int64),
        BufferConfig("rewards", shape=(), dtype=torch.float32),
        BufferConfig("values", shape=(), dtype=torch.float32),
        BufferConfig("policies", shape=(num_actions,), dtype=torch.float32),
        BufferConfig("to_plays", shape=(), dtype=torch.int16),
        BufferConfig("chances", shape=(1,), dtype=torch.int16),
        BufferConfig("game_ids", shape=(), dtype=torch.int64),
        BufferConfig("ids", shape=(), dtype=torch.int64),
        BufferConfig("training_steps", shape=(), dtype=torch.int64),
        BufferConfig("terminated", shape=(), dtype=torch.bool),
        BufferConfig("dones", shape=(), dtype=torch.bool),
        BufferConfig("legal_masks", shape=(num_actions,), dtype=torch.bool),
    ]
    
    # We test just the output processor behavior directly by mocking the buffer dict
    processor = NStepUnrollProcessor(
        unroll_steps=unroll_steps,
        n_step=n_step,
        gamma=gamma,
        num_actions=num_actions,
        num_players=num_players,
        max_size=max_size,
    )
    
    # Create manual buffer dictionaries
    # We sample index 0
    # Steps: 0, 1, 2, 3, 4, 5
    # Step 2 is TERMINATED.
    # Therefore steps 3, 4, 5 belong to a new episode but should be masked.
    game_id_series = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.int64)
    terminated_series = torch.tensor([False, False, True, False, False, False, False, False, False, False], dtype=torch.bool)
    
    buffers = {
        "observations": torch.ones((max_size, 1), dtype=torch.float32),
        "actions": torch.zeros(max_size, dtype=torch.int64),
        "rewards": torch.ones(max_size, dtype=torch.float32),
        "values": torch.ones(max_size, dtype=torch.float32),
        "policies": torch.ones((max_size, num_actions), dtype=torch.float32),
        "to_plays": torch.zeros((max_size), dtype=torch.int64),
        "chances": torch.zeros((max_size, 1), dtype=torch.int64),
        "game_ids": game_id_series.unsqueeze(1),  # Wait, usually game_ids is [B, 1] internally or [B]?
        "ids": torch.arange(max_size, dtype=torch.int64),
        "training_steps": torch.zeros(max_size, dtype=torch.int64),
        "terminated": terminated_series,
        "dones": terminated_series,
        "legal_masks": torch.ones((max_size, num_actions), dtype=torch.bool)
    }
    
    # Fix game_ids shape based on how NStepUnroll Processor expects them
    buffers["game_ids"] = buffers["game_ids"].squeeze(1) if buffers["game_ids"].dim() == 2 else buffers["game_ids"]
    # Wait: NStepUnrollProcessor does: raw_game_ids[:, 0].unsqueeze(1) - it assumes raw_game_ids is [Batch, sequence]
    # Actually `process_batch` takes `indices` and does `raw_game_ids = buffers["game_ids"][all_indices]`
    # where all_indices is [batch_size, unroll_steps + 1]. 
    # Therefore buffers['game_ids'] is `[max_size]` or `[max_size, 1]`. If it's [max_size], all_indices extracts [batch_size, unroll_window].
    # Let's verify what `raw_game_ids[:, 0]` requires -> dim = 2!
    # If buffers["game_ids"] is [max_size], then raw_game_ids is [B, T]. `raw_game_ids[:, 0]` is [B]. `unsqueeze(1)` is [B, 1]. This works perfectly.
    
    batch = processor.process_batch(indices=[0], buffers=buffers)
    
    # Assert sequence shapes
    # unroll_steps=5 means target_policies has shape [Batch, unroll+1, num_actions] = [1, 6, 4]
    B = batch["policies"].shape[0]
    T = batch["policies"].shape[1]
    assert B == 1, "Batch shape should be 1."
    assert T == 6, "Sequence shape (T) should be equal to unroll_steps + 1 (5+1=6)."
    
    # Assert Indices [3:] are padded properly (assuming 0-indexed where 2 is terminating step)
    # The valid dynamic mask (has_valid_action_mask) excludes terminal states.
    # It should be True for step 0 and 1. False for step 2, 3, 4, 5.
    # wait, terminal state is valid to OBSERVE but invalid to act from.
    mask = batch["has_valid_action_mask"][0]
    
    assert mask[0].item() == True
    assert mask[1].item() == True
    assert mask[2].item() == False # Terminal step cannot act
    assert mask[3].item() == False # Post-terminal
    assert mask[4].item() == False
    assert mask[5].item() == False
    
    # Target policies zeroes out padded indices
    # (has_valid_action_mask is matched to target_policies logic in NStepUnrollProcessor)
    # Actually, policies are set identically: target_policies[~is_consistent, u] = 0
    # Wait: in NStepUnrollProcessor `target_policies[is_consistent, u] = raw_p[is_consistent]` and it leaves others at 0.
    policies = batch["policies"][0] # Shape [6, 4]
    
    for u in range(3, 6):
        assert torch.all(policies[u] == 0.0), f"Step {u} policies are not 0.0"
        
    print("Trajectory Padding Contract Test Verified!")
