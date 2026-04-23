import pytest
import torch
from runtime.state import ReplayBuffer
from core.graph import Graph, NODE_TYPE_REPLAY_QUERY
from runtime.executor import execute
from runtime.context import ExecutionContext, ActorSnapshot

pytestmark = pytest.mark.unit

def test_replay_query_filtering():
    """Test filtering by metadata and policy version."""
    rb = ReplayBuffer(capacity=100)
    
    # 1. Add mixed data
    # On-policy data (v1)
    for i in range(5):
        rb.add({
            "obs": torch.tensor([float(i)]),
            "metadata": {
                "on_policy": True,
                "context": {"actor_snapshots": {"actor": {"policy_version": 1}}}
            }
        })
    
    # Expert data (v0)
    for i in range(5):
        rb.add({
            "obs": torch.tensor([float(i + 10)]),
            "metadata": {
                "is_expert": True,
                "context": {"actor_snapshots": {"actor": 0}} # Test old int format too
            }
        })
        
    # 2. Query On-Policy
    graph = Graph()
    graph.add_node("query", NODE_TYPE_REPLAY_QUERY, params={
        "replay_buffer": rb,
        "batch_size": 10,
        "filters": {"on_policy": True}
    })
    
    res = execute(graph, initial_inputs={})
    samples = res["query"].data
    assert len(samples) == 5
    for s in samples:
        assert s["metadata"]["on_policy"] is True

    # 3. Query Expert
    graph_expert = Graph()
    graph_expert.add_node("query", NODE_TYPE_REPLAY_QUERY, params={
        "replay_buffer": rb,
        "batch_size": 10,
        "filters": {"is_expert": True}
    })
    
    res_expert = execute(graph_expert, initial_inputs={})
    samples_expert = res_expert["query"].data
    assert len(samples_expert) == 5
    for s in samples_expert:
        assert s["metadata"]["is_expert"] is True

def test_replay_query_temporal_window():
    """Test temporal window constraint."""
    rb = ReplayBuffer(capacity=100)
    for i in range(10):
        rb.add({"val": torch.tensor([float(i)])})
        
    graph = Graph()
    graph.add_node("query", NODE_TYPE_REPLAY_QUERY, params={
        "replay_buffer": rb,
        "batch_size": 5,
        "temporal_window": 3
    })
    
    res = execute(graph, initial_inputs={})
    samples = res["query"].data
    assert len(samples) == 3
    # Should be the last 3 added (7, 8, 9)
    vals = sorted([s["val"].item() for s in samples])
    assert vals == [7.0, 8.0, 9.0]

def test_replay_query_contiguous():
    """Test contiguous sampling."""
    rb = ReplayBuffer(capacity=100)
    for i in range(10):
        rb.add({"val": torch.tensor([float(i)])})
        
    graph = Graph()
    graph.add_node("query", NODE_TYPE_REPLAY_QUERY, params={
        "replay_buffer": rb,
        "batch_size": 4,
        "contiguous": True
    })
    
    res = execute(graph, initial_inputs={})
    samples = res["query"].data
    assert len(samples) == 4
    # Check if contiguous
    first_val = samples[0]["val"].item()
    for i in range(4):
        assert samples[i]["val"].item() == first_val + i

def test_replay_query_version_constraint():
    """Test filtering by policy version."""
    rb = ReplayBuffer(capacity=100)
    # v1 data
    rb.add({"metadata": {"context": {"actor_snapshots": {"actor": 1}}}})
    # v2 data
    rb.add({"metadata": {"context": {"actor_snapshots": {"actor": 2}}}})
    
    graph = Graph()
    graph.add_node("query", NODE_TYPE_REPLAY_QUERY, params={
        "replay_buffer": rb,
        "batch_size": 10,
        "filters": {"policy_version": 2}
    })
    
    res = execute(graph, initial_inputs={})
    samples = res["query"].data
    assert len(samples) == 1
    # Check version
    snap = samples[0]["metadata"]["context"]["actor_snapshots"]["actor"]
    ver = snap["policy_version"] if isinstance(snap, dict) else snap
    assert ver == 2

def test_replay_query_prefetch():
    """Verify that prefetching respects query parameters."""
    rb = ReplayBuffer(capacity=100)
    for i in range(10):
        rb.add({
            "val": torch.tensor([float(i)]),
            "metadata": {"is_odd": i % 2 != 0}
        })
        
    # Prefetch odd numbers
    import time
    thread = rb.prefetch(batch_size=5, count=1, filters={"is_odd": True})
    thread.join()
    
    # Should get prefetched samples
    samples = rb.sample_query(batch_size=5, filters={"is_odd": True})
    assert len(samples) <= 5
    for s in samples:
        assert s["metadata"]["is_odd"] is True
