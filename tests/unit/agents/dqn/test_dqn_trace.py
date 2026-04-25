import pytest
import torch
from agents.dqn.config import DQNConfig
from agents.dqn.agent import DQNAgent
from runtime.executor import execute

pytestmark = pytest.mark.unit

def test_dqn_trace_readable():
    """
    Verifies that the execution trace of the DQN actor is readable and contains
    the expected nodes.
    """
    config = DQNConfig(obs_dim=4, act_dim=2)
    agent = DQNAgent(config)
    ctx = agent.get_execution_context()
    
    obs = torch.randn(4)
    inputs = {
        "obs_in": obs,
        "clock_in": torch.tensor(0, dtype=torch.int64)
    }
    results = execute(agent.actor_graph, inputs, context=ctx)
    
    # Check for expected nodes in the results
    node_ids = list(results.keys())
    assert "obs_in" in node_ids
    assert "q_values" in node_ids
    assert "epsilon_decay" in node_ids
    assert "actor" in node_ids
    
    # Check that outputs are recorded
    actor_entry = results["actor"]
    assert actor_entry is not None
