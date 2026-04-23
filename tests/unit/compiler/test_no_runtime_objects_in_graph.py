import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from core.graph import Graph, Node
from agents.dqn.agent import DQNAgent
from agents.dqn.config import DQNConfig

pytestmark = pytest.mark.unit

def is_json_serializable(obj):
    """
    Check if an object is JSON-serializable (primitive types).
    We allow lists and dicts of primitives as well.
    """
    if obj is None:
        return True
    if isinstance(obj, (str, int, float, bool)):
        return True
    if isinstance(obj, list):
        return all(is_json_serializable(item) for item in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, str) and is_json_serializable(v) for k, v in obj.items())
    return False

def test_no_runtime_objects_in_dqn_graphs():
    """
    Assert that all node parameters in DQN graphs are JSON-like primitives.
    Specifically reject live objects like torch Modules, Optimizers, or ReplayBuffers.
    """
    config = DQNConfig(
        obs_dim=4,
        act_dim=2,
        hidden_dim=64,
        lr=1e-3,
        buffer_capacity=1000
    )
    agent = DQNAgent(config)
    
    # We check BEFORE and AFTER compilation
    for graph_name, graph in [("actor", agent.actor_graph), ("learner", agent.learner_graph)]:
        for node_id, node in graph.nodes.items():
            for param_name, param_value in node.params.items():
                assert is_json_serializable(param_value), (
                    f"Non-JSON param '{param_name}' in node '{node_id}' ({node.node_type}) "
                    f"of {graph_name} graph. Value type: {type(param_value)}"
                )

    agent.compile(strict=True)

    for graph_name, graph in [("actor", agent.actor_graph), ("learner", agent.learner_graph)]:
        for node_id, node in graph.nodes.items():
            for param_name, param_value in node.params.items():
                assert is_json_serializable(param_value), (
                    f"Non-JSON param '{param_name}' in node '{node_id}' ({node.node_type}) "
                    f"of compiled {graph_name} graph. Value type: {type(param_value)}"
                )

def test_reject_live_objects_explicitly():
    """
    Verify that our check actually catches live objects.
    """
    class MockObj:
        pass

    assert is_json_serializable("hello")
    assert is_json_serializable(123)
    assert is_json_serializable({"a": 1, "b": [True, False]})
    
    assert not is_json_serializable(MockObj())
    assert not is_json_serializable(torch.zeros(1))
    assert not is_json_serializable(nn.Linear(1, 1))
    assert not is_json_serializable(optim.Adam([torch.nn.Parameter(torch.zeros(1))]))
