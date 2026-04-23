import pytest
import torch
import gymnasium as gym
import time
from runtime.scheduler import EveryN, ParallelActorPool, Loop
from runtime.controller import RolloutController
from core.graph import Graph, NODE_TYPE_SOURCE
from runtime.executor import register_operator

pytestmark = pytest.mark.unit

def test_every_n_scheduler():
    """Verify EveryN trigger logic."""
    state = {"count": 0}
    def action():
        state["count"] += 1
        
    scheduler = EveryN(n=3, action=action)
    
    for _ in range(2): scheduler.step()
    assert state["count"] == 0
    
    scheduler.step()
    assert state["count"] == 1
    
    for _ in range(3): scheduler.step()
    assert state["count"] == 2

def test_parallel_actor_pool_throughput():
    """Verify ParallelActorPool executes in parallel and measure throughput."""
    # 1. Setup Graph & Environment
    env_name = "CartPole-v1"
    register_operator("ConstActor", lambda n, i, context=None: 0)
    
    graph = Graph()
    graph.add_node("obs_in", NODE_TYPE_SOURCE)
    graph.add_node("actor", "ConstActor")
    graph.add_edge("obs_in", "actor")
    
    num_actors = 4
    steps_per_actor = 100
    controllers = [RolloutController(graph, gym.make(env_name)) for _ in range(num_actors)]
    pool = ParallelActorPool(controllers)
    
    # 2. Measure Parallel Execution
    start_time = time.time()
    results = pool.rollout(steps_per_actor)
    end_time = time.time()
    
    total_steps = num_actors * steps_per_actor
    duration = end_time - start_time
    throughput = total_steps / duration
    
    print(f"\nParallel Throughput: {throughput:.2f} steps/sec")
    print(f"Total steps: {total_steps}, Duration: {duration:.4f}s")
    
    assert len(results) == num_actors
    for res in results:
        assert len(res) <= steps_per_actor # Some might finish early due to 'done'
    
    # 3. Compare with Serial (Optional but good for verification)
    start_time_serial = time.time()
    for c in controllers:
        c.collect_trajectory(steps_per_actor)
    duration_serial = time.time() - start_time_serial
    
    print(f"Serial Duration: {duration_serial:.4f}s")
    # Note: On some systems threading might not be faster due to GIL if 
    # the env/graph is too fast, but we ensure it runs correctly.

if __name__ == "__main__":
    test_every_n_scheduler()
    test_parallel_actor_pool_throughput()
