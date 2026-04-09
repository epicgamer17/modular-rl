import torch
import numpy as np
import pytest
from typing import Any

from components.environment import SimpleEnvObservationComponent, SimpleEnvStepComponent
from core import Blackboard, BlackboardEngine, infinite_ticks, PipelineComponent

pytestmark = pytest.mark.unit

class FakeEnv:
    """Minimal Gymnasium-like environment for testing."""
    def __init__(self, obs_shape=(4,), max_steps=5):
        self.obs_shape = obs_shape
        self.max_steps = max_steps
        self._step_count = 0

    def reset(self, seed=None):
        self._step_count = 0
        obs = np.ones(self.obs_shape, dtype=np.float32) * 10.0
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = np.ones(self.obs_shape, dtype=np.float32) * (10.0 + self._step_count)
        reward = 1.0
        terminated = self._step_count >= self.max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}

class ConstantActionComponent(PipelineComponent):
    def __init__(self, action: int = 0):
        self.action = action
    def execute(self, blackboard: Blackboard) -> None:
        blackboard.predictions["actions"] = torch.tensor([self.action])

def test_blackboard_loop_parity():
    """
    Verifies that a BlackboardEngine loop produces the same 
    transitions as a standard procedural for-loop.
    """
    device = torch.device("cpu")
    num_steps = 10
    obs_shape = (4,)
    
    # --- 1. Procedural Loop ---
    env_proc = FakeEnv(obs_shape=obs_shape, max_steps=5)
    proc_transitions = []
    
    obs, _ = env_proc.reset()
    for _ in range(num_steps):
        action = 0
        next_obs, reward, terminated, truncated, _ = env_proc.step(action)
        done = terminated or truncated
        
        proc_transitions.append({
            "obs": obs.copy(),
            "action": action,
            "reward": float(reward),
            "next_obs": next_obs.copy(),
            "done": done
        })
        
        if done:
            obs, _ = env_proc.reset()
        else:
            obs = next_obs

    # --- 2. Blackboard Loop ---
    env_bb = FakeEnv(obs_shape=obs_shape, max_steps=5)
    obs_comp = SimpleEnvObservationComponent(env_bb)
    step_comp = SimpleEnvStepComponent(env_bb, obs_comp)
    action_comp = ConstantActionComponent(action=0)
    
    pipeline = [
        obs_comp,
        action_comp,
        step_comp
    ]
    
    engine = BlackboardEngine(components=pipeline, device=device)
    bb_transitions = []
    
    # Create a custom component to capture the blackboard data for testing
    class CaptureComponent(PipelineComponent):
        def execute(self, blackboard: Blackboard) -> None:
            # Deep copy or detach to avoid shared references if needed
            # But here we just want to verify the values
            bb_transitions.append({
                "obs": blackboard.data["observations"].squeeze(0).numpy().copy(),
                "action": blackboard.data["actions"].item(),
                "reward": blackboard.data["rewards"].item(),
                "next_obs": blackboard.data["next_observations"].squeeze(0).numpy().copy(),
                "done": blackboard.data["dones"].item()
            })
            
    pipeline.append(CaptureComponent())
    
    # Run the engine for num_steps
    gen = engine.step(infinite_ticks())
    for _ in range(num_steps):
        next(gen)

    # --- 3. Parity Check ---
    assert len(proc_transitions) == len(bb_transitions)
    
    for i in range(num_steps):
        p = proc_transitions[i]
        b = bb_transitions[i]
        
        assert np.allclose(p["obs"], b["obs"]), f"Obs mismatch at step {i}"
        assert p["action"] == b["action"], f"Action mismatch at step {i}"
        assert p["reward"] == b["reward"], f"Reward mismatch at step {i}"
        assert np.allclose(p["next_obs"], b["next_obs"]), f"Next obs mismatch at step {i}"
        assert p["done"] == b["done"], f"Done mismatch at step {i}"

    print("\nParity test passed! Blackboard loop matches procedural loop.")
