import pytest
import torch
import numpy as np
from data.processors.nstep import NStepUnrollProcessor

pytestmark = pytest.mark.unit

def test_muzero_k_plus_1_contract_alignment():
    """
    Strictly verifies the K+1 Indexing Contract for MuZero unrolls.
    
    Contract:
    - Index 0: Root state (s0). Dummy reward/action. Masked.
    - Index u: Unrolled state (su). Transition (au, ru) reaching su. Unmasked.
    """
    K = 5
    N = 3
    batch_size = 2
    max_size = 100
    L = 20
    
    # 1. Setup Mock Buffer with recognizable patterns
    # Global buffers must have shape [Capacity, ...]
    obs = torch.zeros((max_size, 1))
    rew = torch.zeros(max_size)
    val = torch.zeros(max_size)
    act = torch.zeros(max_size, dtype=torch.long)
    to_play = torch.zeros(max_size)
    game_ids = torch.zeros(max_size)
    dones = torch.zeros(max_size, dtype=torch.bool)
    
    for i in range(L):
        obs[i, 0] = i * 1.0  # s0=0.0, s1=1.0, etc.
        rew[i] = (i + 1) * 0.1  # r1=0.1, r2=0.2 (r_i is reward for s_{i-1}->s_i)
        val[i] = i * 10.0   # v0=0.0, v1=10.0
        act[i] = i + 1      # a1=1, a2=2 (a_i reached s_i)
        to_play[i] = i % 2  # Turn for s_i
        
    buffers = {
        "observations": obs,
        "rewards": rew,
        "values": val,
        "actions": act,
        "policies": torch.zeros((max_size, 10)),  # Add missing policies
        "to_plays": to_play,
        "chances": torch.zeros((max_size, 1)),
        "episode_ids": game_ids,
        "step_ids": torch.arange(max_size, dtype=torch.int32),
        "legal_masks": torch.ones((max_size, 10)),
        "dones": dones,  # Add missing dones
        "terminated": dones,
        "truncated": dones,
        "ids": torch.arange(max_size),
        "training_steps": torch.zeros(max_size),
    }
    
    processor = NStepUnrollProcessor(
        unroll_steps=K,
        n_step=N,
        gamma=1.0,
        num_actions=10,
        num_players=2,
        max_size=max_size
    )
    
    # 2. Process indices (start at 0)
    indices = [0, 0]
    batch = processor.process_batch(indices, buffers)
    
    # 3. Verify K+1 Length
    T = K + 1
    assert batch["values"].shape[1] == T
    assert batch["rewards"].shape[1] == T
    assert batch["actions"].shape[1] == T
    assert batch["reward_mask"].shape[1] == T
    
    # 4. Verify Root Index (u=0) Semantics
    # Observations: observations[0] should be s0
    assert torch.all(batch["unroll_observations"][:, 0] == 0.0)
    
    # Values: values[0] should be v0 (or n-step calc of it)
    # Mover logic at s0:
    # - r1 (s0->s1, mover 0): +0.1
    # - r2 (s1->s2, mover 1): -0.2
    # - r3 (s2->s3, mover 0): +0.3
    # - v3 (at s3, mover 1): -30.0
    # v0_target = 0.1 - 0.2 + 0.3 - 30.0 = -29.8
    assert torch.allclose(batch["values"][:, 0], torch.tensor(-29.8))
    
    # REWARDS/ACTIONS: Index 0 must be DUMMY
    assert torch.all(batch["rewards"][:, 0] == 0.0), "rewards[0] must be dummy 0.0"
    assert torch.all(batch["actions"][:, 0] == 0), "actions[0] must be dummy 0"
    
    # MASKS: Index 0 must be FALSE
    assert torch.all(batch["reward_mask"][:, 0] == False), "reward_mask[0] must be False"
    assert torch.all(batch["to_play_mask"][:, 0] == False), "to_play_mask[0] must be False"
    
    # 5. Verify Unroll Index (u=1) Semantics
    # u=1 maps to s1, r1, a1
    # Note: rewards[1] is reward for reaching s1 (from s0).
    assert torch.all(batch["unroll_observations"][:, 1] == 1.0)
    assert torch.allclose(batch["rewards"][:, 1], torch.tensor(0.1)), "rewards[1] should be r1 (0.1)"
    assert torch.all(batch["actions"][:, 1] == 1), "actions[1] should be a1 (1)"
    
    # Verify values[1] target starting from s1 (mover 1):
    # - r2 (s1->s2, mover 1): +0.2
    # - r3 (s2->s3, mover 0): -0.3
    # - r4 (s3->s4, mover 1): +0.4
    # - v4 (at s4, mover 0): -40.0
    # v1_target = 0.2 - 0.3 + 0.4 - 40.0 = -39.7
    assert torch.allclose(batch["values"][:, 1], torch.tensor(-39.7))
    
    # 6. Verify Terminal Handling (Force terminal at index 2)
    dones[:] = False 
    dones[2] = True  # s2 is terminal. Transitions from s2 are invalid.
    buffers["terminated"] = dones 
    buffers["dones"] = dones 
    
    batch_term = processor.process_batch(indices, buffers)
    
    # Value grounding: V(terminal state) should be 0
    # s2 is terminal. So values[2] MUST be 0.0.
    assert torch.all(batch_term["values"][:, 2] == 0.0), "Terminal value should be grounded"
    
    # Value[0] (s0) with terminal at s2:
    # - r1 (s0->s1, mover 0): +0.1
    # - r2 (s1->s2, mover 1): -0.2 (REACHED TERMINAL)
    # v0_target = 0.1 - 0.2 = -0.1
    assert torch.allclose(batch_term["values"][:, 0], torch.tensor(-0.1))

    # Value[1] (s1) with terminal at s2:
    # - r2 (s1->s2, mover 1): +0.2 (REACHED TERMINAL)
    # v1_target = 0.2
    assert torch.allclose(batch_term["values"][:, 1], torch.tensor(0.2))
    
    # Final check of the key names in return dict
    assert "dones" in batch_term
    
    print("MuZero Alignment Contract Test Passed!")

if __name__ == "__main__":
    test_muzero_k_plus_1_contract_alignment()
