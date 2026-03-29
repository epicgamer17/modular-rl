import pytest
import torch
import numpy as np
from agents.learner.batch_iterators import PPOEpochIterator
from replay_buffers.modular_buffer import ModularReplayBuffer, BufferConfig
from replay_buffers.processors import PPOBatchProcessor

pytestmark = pytest.mark.integration

def test_ppo_advantage_normalization_flow():
    """
    Tier 2: Integration Test.
    Verifies the flow of advantages from the ReplayBuffer (rollout-level norm)
    to the PPOEpochIterator (mini-batch level norm).
    Checks that advantages are ONLY normalized at the mini-batch level.
    """
    torch.manual_seed(42)
    
    # 1. Setup buffer with PPOBatchProcessor (Rollout level)
    max_size = 8
    config = [
        BufferConfig("observations", shape=(1,), dtype=torch.float32),
        BufferConfig("actions", shape=(), dtype=torch.int64),
        BufferConfig("advantages", shape=(), dtype=torch.float32),
        BufferConfig("returns", shape=(), dtype=torch.float32),
        BufferConfig("log_prob", shape=(), dtype=torch.float32),
        BufferConfig("legal_moves_masks", shape=(1,), dtype=torch.bool),
    ]
    buffer = ModularReplayBuffer(
        max_size=max_size, 
        buffer_configs=config, 
        batch_size=max_size,
        output_processor=PPOBatchProcessor()
    )
    
    # 2. Fill with extremely skewed values [1000, 2000, ..., 8000]
    for i in range(max_size):
        buffer.store(
            observations=torch.tensor([0.0]),
            actions=torch.tensor(0),
            advantages=torch.tensor(float(i + 1) * 1000.0),
            returns=torch.tensor(0.0),
            log_prob=torch.tensor(0.0),
            legal_moves_masks=torch.tensor([True])
        )
    
    # 3. Check Rollout-Level Normalization (Buffer output) -> MUST NOT BE NORMALIZED
    full_batch = buffer.sample()
    rollout_adv = full_batch["advantages"]
    
    # Mean of 1000...8000 is 4500.0
    torch.testing.assert_close(rollout_adv.mean(), torch.tensor(4500.0), atol=1e-6, rtol=1e-6)
    
    # 4. Check Mini-Batch Level Normalization (Iterator output) -> MUST BE NORMALIZED
    # 8 samples, 2 minibatches -> 4 samples each.
    # Each mini-batch should be normalized to exactly 0/1.
    iterator = PPOEpochIterator(
        replay_buffer=buffer,
        num_epochs=1,
        num_minibatches=2,
        device=torch.device("cpu"),
        normalize_advantages=True
    )
    
    for mb_idx, minibatch in enumerate(iterator):
        mb_adv = minibatch["advantages"]
        
        # Verify mini-batch level stats
        torch.testing.assert_close(mb_adv.mean(), torch.tensor(0.0), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(mb_adv.std(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)
        
    print("Integration: Advantage Normalization strictly bounded to mini-batch level.")

@pytest.mark.unit
def test_gaeprocessor_integration():
    """
    Tier 1/2: Checks that GAEProcessor produces advantages that PPOBatchProcessor correctly leaves raw.
    """
    from replay_buffers.processors import GAEProcessor, PPOBatchProcessor
    from dataclasses import dataclass

    @dataclass
    class FakeSeq:
        rewards: list
        value_history: list
        observation_history: list
        action_history: list
        log_prob_history: list
        terminated_history: list
        truncated_history: list
        is_terminal: bool = True

    # 1. GAE Processor computes raw advantages
    gae_proc = GAEProcessor(gamma=0.9, gae_lambda=0.9)
    seq = FakeSeq(
        rewards=[1.0, 1.0],
        value_history=[0.5, 0.5],
        observation_history=[np.zeros(1), np.zeros(1)],
        action_history=[0, 0],
        log_prob_history=[0.0, 0.0],
        terminated_history=[False, True],
        truncated_history=[False, False]
    )
    
    # GAEProcessor.process_sequence returns a dict with 'transitions'
    processed = gae_proc.process_sequence(seq)
    raw_advs = [t["advantages"] for t in processed["transitions"]]
    
    assert len(raw_advs) == 2
    # Advantages aren't normalized yet
    assert not np.isclose(np.mean(raw_advs), 0.0, atol=1e-3)
    
    # 2. PPOBatchProcessor (Output) passes them raw
    # Setup mock buffers
    buffers = {
        "advantages": torch.tensor(raw_advs, dtype=torch.float32),
        "observations": torch.zeros(2, 1),
        "actions": torch.zeros(2, dtype=torch.int64),
        "returns": torch.zeros(2),
        "log_prob": torch.zeros(2),
        "legal_moves_masks": torch.zeros(2, 1, dtype=torch.bool)
    }
    
    norm_proc = PPOBatchProcessor()
    out = norm_proc.process_batch(indices=[0, 1], buffers=buffers)
    
    # Still NOT centered at 0
    assert not torch.allclose(out["advantages"].mean(), torch.tensor(0.0), atol=1e-6)
    print("Integration: GAEProcessor -> PPOBatchProcessor raw passthrough confirmed.")

