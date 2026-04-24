import pytest
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from runtime.runtime import ActorRuntime
from runtime.context import ExecutionContext, ActorSnapshot
from runtime.state import ParameterStore
from core.graph import Graph, NODE_TYPE_SOURCE
from runtime.executor import register_operator
from torch.func import functional_call

pytestmark = pytest.mark.unit

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0]))
    def forward(self, x):
        return x * self.w

class MockEnv:
    def __init__(self):
        self.reset_called = False
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.action_space = gym.spaces.Discrete(2)
        
    def reset(self, seed=None):
        self.reset_called = True
        return np.array([1.0], dtype=np.float32), {}
        
    def step(self, action):
        return np.array([1.0], dtype=np.float32), 0.0, False, False, {}

def test_actor_snapshot_immutability():
    """
    Test 12.2: Ensure actor continues using frozen snapshot even if state is modified.
    """
    # 1. Setup
    model = SimpleModel()
    ps = ParameterStore(dict(model.named_parameters()))
    
    graph = Graph()
    graph.add_node("obs_in", NODE_TYPE_SOURCE)
    
    def op_actor(node, inputs, context=None):
        obs = inputs["obs"]
        snapshot = context.get_actor_snapshot(node.node_id)
        # Use functional call with snapshot state
        out = functional_call(model, snapshot.state, (obs,))
        return out

    register_operator("SnapshotActor", op_actor)
    graph.add_node("actor", "SnapshotActor", params={"param_store": ps})
    graph.add_edge("obs_in", "actor", dst_port="obs")
    
    env = MockEnv()
    runtime = ActorRuntime(graph, env)
    
    # 2. First Step: Should create snapshot v0 (w=1.0)
    ctx = ExecutionContext()
    obs = torch.tensor([10.0])
    runtime.current_obs = obs # Manual inject for test
    
    step_1 = runtime.step(ctx)
    assert step_1["action"] == 10.0
    assert ctx.get_actor_snapshot("actor").policy_version == 0
    
    # 3. Modify Parameters in ParameterStore
    ps.update_state({"w": torch.tensor([2.0])})
    assert ps.version == 1
    
    # 4. Second Step with SAME Context: Should still use snapshot v0 (w=1.0)
    runtime.current_obs = obs # Re-inject same obs
    step_2 = runtime.step(ctx)
    assert step_2["action"] == 10.0, f"Expected 10.0 (frozen), got {step_2['action']}"
    
    # 5. New Context: Should create NEW snapshot v1 (w=2.0)
    ctx_new = ExecutionContext()
    runtime.current_obs = obs # Reset for next step simulation
    step_3 = runtime.step(ctx_new)
    assert step_3["action"] == 20.0, f"Expected 20.0 (new), got {step_3['action']}"
    
    print("Actor Snapshot Immutability Verified.")

def test_manual_snapshot_binding():
    """Verify that manually bound snapshots are respected."""
    model = SimpleModel()
    ps = ParameterStore(dict(model.named_parameters()))
    
    graph = Graph()
    graph.add_node("obs_in", NODE_TYPE_SOURCE)
    
    def manual_op(node, inputs, context=None):
        snap = context.get_actor_snapshot(node.node_id)
        return functional_call(model, snap.state, (inputs["obs"],))

    register_operator("SnapshotActorManual", manual_op)
    graph.add_node("actor", "SnapshotActorManual", params={"param_store": ps})
    graph.add_edge("obs_in", "actor", dst_port="obs")
    
    env = MockEnv()
    runtime = ActorRuntime(graph, env)
    
    # Manually bind a "fake" snapshot
    fake_state = {"w": torch.tensor([5.0])}
    fake_snapshot = ActorSnapshot(policy_version=99, state=fake_state)
    
    ctx = ExecutionContext()
    ctx.bind_actor("actor", fake_snapshot)
    
    runtime.current_obs = torch.tensor([1.0])
    step = runtime.step(ctx)
    
    assert step["action"] == 5.0
    
    print("Manual Snapshot Binding Verified.")

def test_snapshot_includes_buffers():
    """Verify that snapshots include both parameters and buffers."""
    class ModelWithBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([1.0]))
            self.register_buffer("b", torch.tensor([0.5]))
        def forward(self, x):
            return x * self.w + self.b

    model = ModelWithBuffer()
    state = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    ps = ParameterStore(state)
    
    ctx = ExecutionContext()
    snap = ActorSnapshot(policy_version=1, state=ps.get_state())
    
    assert "w" in snap.state
    assert "b" in snap.state
    assert snap.state["b"] == 0.5
    
    # Verify functional call works with both
    out = functional_call(model, snap.state, (torch.tensor([1.0]),))
    assert out == 1.5
