import pytest
import torch
import numpy as np
from core.graph import Graph, NODE_TYPE_SOURCE, NODE_TYPE_EXPLORATION
from agents.dqn.agent import DQNAgent
from agents.dqn.config import DQNConfig
from runtime.executor import execute
from runtime.context import ExecutionContext
from compiler.pipeline import compile_graph

pytestmark = pytest.mark.unit

def test_graph_serialization_roundtrip():
    """
    Test 3: Graph Serialization Roundtrip.
    Verifies: compile -> to_dict -> from_dict -> execute produces identical results.
    """
    # 1. Setup a standard DQN actor graph
    config = DQNConfig(obs_dim=4, act_dim=2)
    agent = DQNAgent(config)
    original_graph = agent.actor_graph
    
    # 2. Compile it
    model_handles = {config.model_handle, config.target_handle}
    compiled_graph = compile_graph(original_graph, optimize=True, model_handles=model_handles)
    
    # 3. Serialize to dict
    graph_dict = compiled_graph.to_dict()
    assert isinstance(graph_dict, dict)
    assert "nodes" in graph_dict
    assert "edges" in graph_dict
    
    # 4. Reconstruct from dict
    reconstructed_graph = Graph.from_dict(graph_dict)
    
    # 5. Verify structure
    assert len(reconstructed_graph.nodes) == len(compiled_graph.nodes)
    assert len(reconstructed_graph.edges) == len(compiled_graph.edges)
    
    # 6. Execute both and compare
    obs = torch.randn(4)
    inputs = {
        "obs_in": obs,
        "clock_in": torch.tensor(0, dtype=torch.int64)
    }
    
    # Same seed for determinism in exploration
    ctx_orig = ExecutionContext(seed=42, model_registry=agent.model_registry)
    results_orig = execute(compiled_graph, inputs, context=ctx_orig)
    
    ctx_recon = ExecutionContext(seed=42, model_registry=agent.model_registry)
    results_recon = execute(reconstructed_graph, inputs, context=ctx_recon)
    
    # Compare outputs of key nodes
    for nid in results_orig:
        if nid in results_recon:
            orig_val = results_orig[nid].data
            recon_val = results_recon[nid].data
            if isinstance(orig_val, torch.Tensor):
                assert torch.allclose(orig_val, recon_val)
            else:
                assert orig_val == recon_val

def test_schema_serialization():
    """Verify that schemas are correctly preserved through serialization."""
    from core.schema import Schema, Field, TensorSpec
    
    spec = TensorSpec(shape=(4,), dtype="float32", tags=["obs"])
    field = Field(name="obs", spec=spec)
    schema = Schema(fields=[field])
    
    graph = Graph()
    graph.add_node("src", NODE_TYPE_SOURCE, schema_out=schema)
    
    recon_graph = Graph.from_dict(graph.to_dict())
    recon_node = recon_graph.nodes["src"]
    
    assert recon_node.schema_out.fields[0].name == "obs"
    assert recon_node.schema_out.fields[0].spec.shape == (4,)
    assert recon_node.schema_out.fields[0].spec.dtype == "float32"
    assert "obs" in recon_node.schema_out.fields[0].spec.tags
