import pytest
import torch
from agents.dqn.config import DQNConfig
from agents.dqn.agent import DQNAgent
from runtime.bootstrap import bootstrap_runtime
from runtime.registry import clear_registry

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def setup():
    clear_registry()
    bootstrap_runtime()


def test_dqn_strict_compile():
    """
    Verifies that the DQN actor and learner graphs pass strict compilation.
    This ensures all nodes are fully typed and semantics are valid.
    """
    config = DQNConfig(obs_dim=4, act_dim=2)
    agent = DQNAgent(config)
    
    agent.compile(strict=True)

    assert agent.actor_graph is not None
    assert agent.learner_graph is not None

def test_dqn_graph_separation():
    """
    Verifies that actor and learner graphs are distinct and correctly structured.
    """
    config = DQNConfig(obs_dim=4, act_dim=2)
    agent = DQNAgent(config)
    
    # Actor graph should have QValuesSingle, LinearDecay, and Exploration
    node_types_actor = [n.node_type for n in agent.actor_graph.nodes.values()]
    assert "QValuesSingle" in node_types_actor
    assert "LinearDecay" in node_types_actor
    assert "Exploration" in node_types_actor
    
    # Learner graph should have ReplayQuery, TDLoss, Backward, AccumulateGrad,
    # OptimizerStepEvery, and TargetSync
    node_types_learner = [n.node_type for n in agent.learner_graph.nodes.values()]
    assert "ReplayQuery" in node_types_learner
    assert "TDLoss" in node_types_learner
    assert "Backward" in node_types_learner
    assert "AccumulateGrad" in node_types_learner
    assert "OptimizerStepEvery" in node_types_learner
    assert "TargetSync" in node_types_learner
