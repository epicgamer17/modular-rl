import pytest
import torch
import gymnasium as gym
import threading
import time
from runtime.engine import ActorRuntime, LearnerRuntime
from runtime.context import ExecutionContext, ActorSnapshot
from runtime.state import ParameterStore, ReplayBuffer
from core.graph import Graph, NODE_TYPE_SOURCE
from core.schema import TAG_ON_POLICY
from runtime.operator_registry import register_operator

pytestmark = pytest.mark.unit

def test_runtime_split_async_behavior():
    """
    Test 11.2: Verify async behavior with split Runtimes.
    Actor produces data continuously, Learner lags behind.
    """
    # 1. Setup State
    params = {"w": torch.tensor(1.0)}
    ps = ParameterStore(params)
    rb = ReplayBuffer(capacity=1000)
    
    # 2. Actor Runtime Setup
    interact_graph = Graph()
    interact_graph.add_node("obs_in", NODE_TYPE_SOURCE)
    
    def op_actor(node, inputs, context=None):
        if context:
            context.bind_actor(node.node_id, ActorSnapshot(ps.version, ps.get_state()))
        return 0
        
    register_operator("ActorNode", op_actor)
    from runtime.signals import NoOp
    register_operator(NODE_TYPE_SOURCE, lambda n, i, c=None: NoOp())
    
    interact_graph.add_node("actor", "ActorNode")
    interact_graph.add_edge("obs_in", "actor")
    
    def actor_record(step_data):
        rb.add(step_data)
        
    actor_runtime = ActorRuntime(interact_graph, gym.make("CartPole-v1"), recording_fn=actor_record)
    
    # 3. Learner Runtime Setup (PPO-style check)
    train_graph = Graph()
    train_graph.add_node("traj_in", NODE_TYPE_SOURCE)
    
    def op_ppo_learner(node, inputs, context=None):
        batch = inputs["traj_in"]
        snap = batch["metadata"]["context"]["actor_snapshots"]["actor"]
        data_version = snap["policy_version"] if isinstance(snap, dict) else snap
        if data_version != ps.version:
            raise ValueError(f"FORBIDDEN STALENESS: Data v{data_version} != Policy v{ps.version}")
        return "ok"
        
    register_operator("PPOLearner", op_ppo_learner)
    train_graph.add_node("learner", "PPOLearner", tags=[TAG_ON_POLICY])
    train_graph.add_edge("traj_in", "learner", dst_port="traj_in")
    
    learner_runtime = LearnerRuntime(train_graph)
    
    # 4. Async Execution Simulation
    def actor_loop():
        for _ in range(50):
            actor_runtime.step()
            time.sleep(0.001)
            
    actor_thread = threading.Thread(target=actor_loop)
    actor_thread.start()
    
    # Learner lags behind
    time.sleep(0.01)
    
    # Scenario A: Learner and Actor are in sync (Initial)
    batch = rb.sample(1)[0]
    results = learner_runtime.update_step(batch=batch)
    assert results["learner"] == "ok"
    
    # Scenario B: Actor updates policy, making old data stale
    ps.update_state({}) # Increment version
    
    # Wait for more data
    time.sleep(0.01)
    
    # Sample OLD data (from v0) while current policy is v1
    # We find an old entry
    old_entry = None
    for item in rb.buffer:
        snap = item["metadata"]["context"]["actor_snapshots"]["actor"]
        data_version = snap["policy_version"] if isinstance(snap, dict) else snap
        if data_version == 0:
            old_entry = item
            break
            
    if old_entry:
        with pytest.raises(ValueError, match="FORBIDDEN STALENESS"):
            learner_runtime.update_step(batch=old_entry)
            
    actor_thread.join()
    print("Runtime split async behavior verified.")

if __name__ == "__main__":
    test_runtime_split_async_behavior()
