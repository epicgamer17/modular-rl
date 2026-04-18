import torch
import pytest
from core.blackboard import Blackboard
from core.blackboard_engine import apply_updates
from components.telemetry import TelemetryComponent, SequenceTerminatorComponent

pytestmark = pytest.mark.unit

def test_telemetry_scalar_consistency():
    """Verify that the new vectorized TelemetryComponent still works correctly for scalar inputs."""
    telemetry = TelemetryComponent()
    blackboard = Blackboard()
    
    # Step 1
    blackboard.meta["reward"] = 1.0
    blackboard.meta["dones"] = False
    outputs = telemetry.execute(blackboard)
    apply_updates(blackboard, outputs)
    
    assert blackboard.meta["running_reward"] == 1.0
    assert blackboard.meta["running_length"] == 1.0
    assert "score" not in blackboard.meta
    
    # Step 2 (Finish)
    blackboard.meta["reward"] = 2.0
    blackboard.meta["dones"] = True
    outputs = telemetry.execute(blackboard)
    apply_updates(blackboard, outputs)
    
    assert blackboard.meta["score"] == 3.0
    assert blackboard.meta["episode_length"] == 2.0
    assert blackboard.meta["running_reward"] == 0.0 # Reset after done
    assert blackboard.meta["running_length"] == 0.0

def test_telemetry_vectorized_reward():
    """Verify that TelemetryComponent handles [B] tensors correctly."""
    B = 2
    telemetry = TelemetryComponent()
    blackboard = Blackboard()
    
    # Step 1: Both continue
    blackboard.meta["reward"] = torch.tensor([1.0, 10.0])
    blackboard.meta["dones"] = torch.tensor([False, False])
    outputs = telemetry.execute(blackboard)
    apply_updates(blackboard, outputs)
    
    assert blackboard.meta["running_reward"] == 5.5  # (1 + 10) / 2
    
    # Step 2: Env 0 finishes, Env 1 continues
    blackboard.meta["reward"] = torch.tensor([1.0, 10.0])
    blackboard.meta["dones"] = torch.tensor([True, False])
    outputs = telemetry.execute(blackboard)
    apply_updates(blackboard, outputs)
    
    # Env 0 finished with 2.0. Env 1 is at 20.0
    assert blackboard.meta["score"] == 2.0
    assert blackboard.meta["episode_length"] == 2
    assert blackboard.meta["running_reward"] == 10.0 # (0 + 20) / 2
    
    # Step 3: Env 1 finishes
    blackboard.meta["reward"] = torch.tensor([1.0, 10.0])
    blackboard.meta["dones"] = torch.tensor([False, True])
    outputs = telemetry.execute(blackboard)
    apply_updates(blackboard, outputs)
    
    # Env 0 started new episode (reward=1.0). Env 1 finished with 30.0
    assert blackboard.meta["score"] == 30.0
    assert blackboard.meta["episode_length"] == 3
    assert blackboard.meta["running_reward"] == 0.5 # (1.0 + 0) / 2

def test_telemetry_universal_time_mandate():
    """Verify that TelemetryComponent handles [B, T] tensors correctly."""
    B, T = 2, 3
    telemetry = TelemetryComponent()
    blackboard = Blackboard()
    
    # reward [2, 3]
    blackboard.meta["reward"] = torch.ones((B, T))
    # done [2, 3] - Env 0 finishes at T=1, Env 1 never finishes
    dones = torch.zeros((B, T), dtype=torch.bool)
    dones[0, 1] = True
    blackboard.meta["dones"] = dones
    
    outputs = telemetry.execute(blackboard)
    apply_updates(blackboard, outputs)
    
    # Total rewards for step: Env 0 = 3.0, Env 1 = 3.0
    # But Env 0 finished, so score should be reported.
    # Note: If it finishes at T=1, the reward summed for the WHOLE T=3 block is 3.0.
    # This is correct for "Universal Time Mandate" where a batch might be a full rollout.
    
    assert blackboard.meta["score"] == 3.0
    assert blackboard.meta["episode_length"] == 3.0

def test_sequence_terminator_vectorized():
    """Verify SequenceTerminatorComponent handles any() correctly."""
    terminator = SequenceTerminatorComponent()
    blackboard = Blackboard()
    
    # Scalar False
    blackboard.meta["dones"] = False
    outputs = terminator.execute(blackboard)
    apply_updates(blackboard, outputs)
    assert not blackboard.meta.get("stop_execution")
    
    # Vector False
    blackboard.meta["dones"] = torch.tensor([False, False])
    outputs = terminator.execute(blackboard)
    apply_updates(blackboard, outputs)
    assert not blackboard.meta.get("stop_execution")
    
    # Vector True
    blackboard.meta["dones"] = torch.tensor([False, True])
    outputs = terminator.execute(blackboard)
    apply_updates(blackboard, outputs)
    assert blackboard.meta["stop_execution"] is True
