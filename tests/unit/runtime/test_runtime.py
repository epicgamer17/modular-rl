import pytest
import torch
import gymnasium as gym
from runtime.runtime import ActorRuntime, LearnerRuntime
from core.graph import Graph, NODE_TYPE_SOURCE
from runtime.executor import register_operator
from runtime.state import ReplayBuffer

pytestmark = pytest.mark.unit

def test_actor_runtime_basic():
    """Verify basic actor rollout orchestration."""
    # 1. Setup mock environment and graph
    env = gym.make("CartPole-v1")
    graph = Graph()
    graph.add_node("obs_in", NODE_TYPE_SOURCE)
    # Simple constant actor
    register_operator("ConstActor", lambda n, i, context=None: 0)
    graph.add_node("actor", "ConstActor")
    graph.add_edge("obs_in", "actor")
    
    # 2. Setup Runtime
    recorded_steps = []
    def record_fn(data):
        recorded_steps.append(data)
        
    runtime = ActorRuntime(graph, env, recording_fn=record_fn)
    
    # 3. Perform steps
    step_data = runtime.step()
    
    assert "obs" in step_data
    assert "action" in step_data
    assert step_data["action"] == 0
    assert len(recorded_steps) == 1
    assert "metadata" in recorded_steps[0]
    assert recorded_steps[0]["metadata"]["step_index"] == 0

def test_dagger_triviality():
    """
    Demonstrates how DAgger (Dataset Aggregation) is simplified by ActorRuntime.
    DAgger requires running one policy (student) but recording another policy's action (expert).
    """
    env = gym.make("CartPole-v1")
    sl_buffer = ReplayBuffer(capacity=100)
    
    # Graph with two actors: student (drives env) and expert (provides labels)
    graph = Graph()
    graph.add_node("obs_in", NODE_TYPE_SOURCE)
    register_operator("StudentActor", lambda n, i, context=None: 0) # Const 0
    register_operator("ExpertActor", lambda n, i, context=None: 1)  # Const 1
    
    graph.add_node("actor", "StudentActor") # The node ID 'actor' is used by runtime for env.step
    graph.add_node("expert", "ExpertActor")
    
    graph.add_edge("obs_in", "actor")
    graph.add_edge("obs_in", "expert")
    
    # DAgger Recording Logic: Record Student's Obs and Expert's Action
    def dagger_record_fn(step_data):
        expert_val = step_data["metadata"]["actor_results"]["expert"]
        sl_buffer.add({
            "obs": step_data["obs"],
            "action": torch.tensor(expert_val.data)
        })
        
    runtime = ActorRuntime(graph, env, recording_fn=dagger_record_fn)
    
    # Execute rollout
    runtime.step()
    
    # Verify SL Buffer has expert's action (1) for student's observation
    assert len(sl_buffer) == 1
    assert sl_buffer.buffer[0]["action"].item() == 1
    print("DAgger triviality verified.")

def test_learner_runtime_basic():
    """Verify basic learner optimization step."""
    train_graph = Graph()
    train_graph.add_node("traj_in", NODE_TYPE_SOURCE)
    
    def mock_opt(node, inputs, context=None):
        return "updated"
        
    register_operator("MockOpt", mock_opt)
    train_graph.add_node("opt", "MockOpt")
    train_graph.add_edge("traj_in", "opt")
    
    from runtime.context import ExecutionContext
    ctx = ExecutionContext()
    learner = LearnerRuntime(train_graph)
    results = learner.update_step(batch={"dummy": "data"}, context=ctx)
    
    assert results["opt"] == "updated"
    assert ctx.learner_step == 1
