import pytest
from core.graph import Graph
from agents.dqn.specs import register_dqn_specs
from compiler.passes.collect_trainable_parameters import collect_trainable_parameters

pytestmark = pytest.mark.unit

def test_collect_dqn_params():
    """Verifies that DQN parameter handles are correctly collected."""
    register_dqn_specs()
    g = Graph()
    # DQN uses 'model_handle' for Q-networks
    g.add_node("q_node", "QValuesSingle", params={"model_handle": "online_q"})
    g.add_node("target_node", "QValuesSingle", params={"model_handle": "target_q"})
    
    params = collect_trainable_parameters(g)
    assert "online_q" in params
    assert "target_q" in params
    assert "q_node" in params["online_q"]
    assert "target_node" in params["target_q"]
