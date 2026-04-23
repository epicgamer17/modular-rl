import pytest
import torch
import torch.nn as nn
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
    def reset(self, seed=None):
        self.reset_called = True
        return torch.tensor([1.0]), {}
    def step(self, action):
        return torch.tensor([1.0]), 0.0, False, False, {}

def test_actor_snapshot_immutability():
    """
    Test 12.2: Ensure actor continues using frozen snapshot even if parameters are modified.
    """
    # 1. Setup
    model = SimpleModel()
    ps = ParameterStore(dict(model.named_parameters()))
    
    graph = Graph()
    graph.add_node("obs_in", NODE_TYPE_SOURCE)
    
    def op_actor(node, inputs, context=None):
        obs = list(inputs.values())[0]
        snapshot = context.get_actor_snapshot(node.node_id)
        # Use functional call with snapshot parameters
        out = functional_call(model, snapshot.parameters, (obs,))
        return out

    register_operator("SnapshotActor", op_actor)
    graph.add_node("actor", "SnapshotActor", params={"param_store": ps})
    graph.add_edge("obs_in", "actor")
    
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
    ps.update_parameters({"w": torch.tensor([2.0])})
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
        return functional_call(model, snap.parameters, (list(inputs.values())[0],))

    register_operator("SnapshotActorManual", manual_op)
    graph.add_node("actor", "SnapshotActorManual", params={"param_store": ps})
    graph.add_edge("obs_in", "actor")
    
    env = MockEnv()
    runtime = ActorRuntime(graph, env)
    
    # Manually bind a "fake" snapshot
    fake_params = {"w": torch.tensor([5.0])}
    fake_snapshot = ActorSnapshot(policy_version=99, parameters=fake_params)
    
    ctx = ExecutionContext()
    ctx.bind_actor("actor", fake_snapshot)
    
    runtime.current_obs = torch.tensor([1.0])
    step = runtime.step(ctx)
    
    assert step["action"] == 5.0
    
    print("Manual Snapshot Binding Verified.")
