import pytest
from core.graph import Graph
from agents.ppo.specs import register_ppo_specs
from compiler.passes.optimization.parameters import collect_trainable_parameters

pytestmark = pytest.mark.unit

def test_collect_ppo_actorcritic_params():
    """Verifies that PPO parameter handles are correctly collected."""
    register_ppo_specs()
    g = Graph()
    # PPO uses 'model_handle' for its ActorCritic network
    g.add_node("actor", "PPO_PolicyActor", params={"model_handle": "ppo_net"})
    g.add_node("ppo_loss", "PPO_Objective", params={"model_handle": "ppo_net"})
    
    params = collect_trainable_parameters(g)
    assert "ppo_net" in params
    assert "actor" in params["ppo_net"]
    assert "ppo_loss" in params["ppo_net"]
