import pytest
import torch
import numpy as np
from agents.dqn.config import DQNConfig
from agents.dqn.agent import DQNAgent
from compiler.compiler import compile_graph
from runtime.executor import execute
from runtime.context import ExecutionContext

pytestmark = pytest.mark.unit

def test_dqn_actor_equivalence():
    """
    Verifies that the original and optimized actor graphs produce the same output
    for the same input and RNG state.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    config = DQNConfig(obs_dim=4, act_dim=2)
    agent = DQNAgent(config)
    ctx = agent.get_execution_context()
    # Context is initialized with seed 42 by default, but we can recreate it to be sure
    ctx = ExecutionContext(seed=42, model_registry=agent.model_registry, buffer_registry=agent.buffer_registry)
    
    obs = torch.randn(4)
    inputs = {
        "obs_in": obs,
        "clock_in": torch.tensor(0, dtype=torch.int64)
    }
    
    # 1. Execute original graph
    original_results = execute(agent.actor_graph, inputs, context=ctx)
    original_action = original_results["actor"].data
    
    # 2. Reset context for optimized graph
    ctx = ExecutionContext(seed=42, model_registry=agent.model_registry, buffer_registry=agent.buffer_registry)
    model_handles = {config.model_handle, config.target_handle}
    optimized_graph = compile_graph(agent.actor_graph, optimize=True, model_handles=model_handles)
    optimized_results = execute(optimized_graph, inputs, context=ctx)
    
    # Find the node that produces the action in the optimized graph
    # (It might have a different ID if it was fused)
    optimized_action = None
    for nid, output in optimized_results.items():
        if nid == "actor" or "fused" in nid:
            if hasattr(output, "data") and isinstance(output.data, int):
                optimized_action = output.data
                break
    
    assert original_action == optimized_action, \
        f"Optimized graph produced different action: {original_action} vs {optimized_action}"
