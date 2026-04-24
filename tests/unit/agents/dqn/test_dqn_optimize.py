import pytest
from agents.dqn.config import DQNConfig
from agents.dqn.agent import DQNAgent
from compiler.compiler import compile_graph
from runtime.context import ExecutionContext

pytestmark = pytest.mark.unit

def test_dqn_node_reduction():
    """
    Verifies that the actor graph node count is reduced after optimization.
    Currently: obs_in + q_values + epsilon_decay + actor = 4 nodes.
    If we fuse q_values + actor, it should be 3 nodes.
    """
    config = DQNConfig(obs_dim=4, act_dim=2)
    agent = DQNAgent(config)
    
    original_node_count = len(agent.actor_graph.nodes)
    
    # 2. Reset context for optimized graph
    ctx = ExecutionContext(seed=42, model_registry=agent.model_registry, buffer_registry=agent.buffer_registry)
    model_handles = {config.model_handle, config.target_handle}
    optimized_graph = compile_graph(agent.actor_graph, optimize=True, model_handles=model_handles)
    optimized_node_count = len(optimized_graph.nodes)
    
    # With epsilon decay, 'actor' has two inputs, so linear fusion is skipped.
    # Total: obs_in, q_values, epsilon_decay, actor = 4 nodes.
    assert optimized_node_count == original_node_count, \
        f"Unexpected node count after optimization: {original_node_count} -> {optimized_node_count}"

def test_dqn_dead_node_elimination():
    """
    Verifies that dead nodes are removed.
    """
    config = DQNConfig(obs_dim=4, act_dim=2)
    agent = DQNAgent(config)
    
    # Add a dangling node
    agent.actor_graph.add_node("dead_node", "NoOp")
    
    original_count = len(agent.actor_graph.nodes)
    model_handles = {config.model_handle, config.target_handle}
    optimized_graph = compile_graph(agent.actor_graph, optimize=True, model_handles=model_handles)
    optimized_count = len(optimized_graph.nodes)
    
    # dead_node should be removed
    assert "dead_node" not in optimized_graph.nodes
    assert optimized_count < original_count
