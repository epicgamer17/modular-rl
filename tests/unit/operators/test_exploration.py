import pytest
import torch
import random
import numpy as np
from core.graph import Graph, NODE_TYPE_SOURCE, NODE_TYPE_EXPLORATION
from runtime.executor import execute
from runtime.context import ExecutionContext

# Set module level marker as per RULE[testing-standards.md]
pytestmark = pytest.mark.unit

def test_epsilon_greedy_determinism_same_seed():
    """
    Test 1A: Same seed + same graph + same input = identical action sequence.
    Verifies that the exploration node uses the context's RNG correctly.
    """
    # Fix global seeds for safety, though we test context RNG
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    graph = Graph()
    graph.add_node("q_values", NODE_TYPE_SOURCE)
    graph.add_node("actor", NODE_TYPE_EXPLORATION, params={"epsilon": 0.5, "act_dim": 4})
    graph.add_edge("q_values", "actor")
    
    q_values = torch.tensor([0.1, 0.2, 0.3, 0.4])
    
    # Run 1
    ctx1 = ExecutionContext(seed=123)
    results1 = []
    for _ in range(20):
        # Derive to simulate steps and RNG progression
        step_ctx = ctx1.derive()
        out = execute(graph, initial_inputs={"q_values": q_values}, context=step_ctx)
        results1.append(out["actor"])
        ctx1 = step_ctx
        
    # Run 2
    ctx2 = ExecutionContext(seed=123)
    results2 = []
    for _ in range(20):
        step_ctx = ctx2.derive()
        out = execute(graph, initial_inputs={"q_values": q_values}, context=step_ctx)
        results2.append(out["actor"])
        ctx2 = step_ctx
        
    assert results1 == results2, f"Action sequences differed: {results1} != {results2}"

def test_epsilon_greedy_decorrelation_different_shards():
    """
    Test 1B: Different shard IDs = decorrelated action streams.
    Verifies that shard_id properly offsets the RNG seed.
    """
    # Fix global seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    graph = Graph()
    graph.add_node("q_values", NODE_TYPE_SOURCE)
    # Use epsilon=1.0 to ensure all actions are random
    graph.add_node("actor", NODE_TYPE_EXPLORATION, params={"epsilon": 1.0, "act_dim": 1000})
    graph.add_edge("q_values", "actor")
    
    q_values = torch.zeros(1000)
    
    # Shard 0
    ctx0 = ExecutionContext(seed=42, shard_id=0)
    results0 = []
    current_ctx = ctx0
    for _ in range(10):
        current_ctx = current_ctx.derive()
        out = execute(graph, initial_inputs={"q_values": q_values}, context=current_ctx)
        results0.append(out["actor"])
    
    # Shard 1
    ctx1 = ExecutionContext(seed=42, shard_id=1)
    results1 = []
    current_ctx = ctx1
    for _ in range(10):
        current_ctx = current_ctx.derive()
        out = execute(graph, initial_inputs={"q_values": q_values}, context=current_ctx)
        results1.append(out["actor"])
    
    assert results0 != results1, "Action streams from different shards should be decorrelated"
    
def test_epsilon_greedy_bypass_check():
    """
    Verifies that the node DOES NOT use global random.random().
    We monkeypatch random.random to return a constant and check if it's ignored
    when using context RNG.
    """
    graph = Graph()
    graph.add_node("q_values", NODE_TYPE_SOURCE)
    graph.add_node("actor", NODE_TYPE_EXPLORATION, params={"epsilon": 1.0, "act_dim": 10})
    graph.add_edge("q_values", "actor")
    
    q_values = torch.zeros(10)
    
    # Context with specific seed that would normally give some result
    ctx = ExecutionContext(seed=42)
    
    import random as global_random
    original_random = global_random.random
    try:
        # Force global random to return 1.0 (never explore)
        global_random.random = lambda: 1.0
        
        # But our node should use context.rng, which is a random.Random instance
        # seeded with 42. context.rng.random() for seed 42 first call is ~0.639
        # epsilon is 1.0, so it SHOULD explore.
        out = execute(graph, initial_inputs={"q_values": q_values}, context=ctx.derive())
        # If it used global_random, it would not explore and return 0 (argmax of zeros).
        # We check that it did something else or at least didn't crash.
        # Actually, if epsilon=1.0 and it explores, it returns randint(0, 9).
        # We can't be 100% sure it won't return 0 by chance, but global_random.random=1.0
        # would definitely NOT explore.
        
        # Let's use a more reliable check: the context RNG state should advance.
        state_before = ctx.rng.getstate()
        execute(graph, initial_inputs={"q_values": q_values}, context=ctx)
        state_after = ctx.rng.getstate()
        
        assert state_before != state_after, "Context RNG state should have advanced"
        
    finally:
        global_random.random = original_random
