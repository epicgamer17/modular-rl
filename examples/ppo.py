from rlir.core.graph import GraphDef
from rlir.core.schema import SchemaDef

def ppo_graph():
    graph = GraphDef("PPO_Actor_Critic")
    
    # Schemas
    obs_schema = SchemaDef("Observation", {"obs": "Float[4]"})
    action_schema = SchemaDef("Action", {"action": "Int"})
    graph.add_schema(obs_schema)
    graph.add_schema(action_schema)
    
    # Nodes
    graph.add_source("env", "CartPole")
    graph.add_actor("policy", "PPOActor")
    graph.add_node("value_net", "ValueFunction", inputs=["in"], outputs=["value"])
    graph.add_node("advantage", "GAE", inputs=["rewards", "values"], outputs=["adv"])
    graph.add_sink("replay", "PPOBuffer")
    
    # Edges
    graph.add_edge("env", "policy")
    graph.add_edge("env", "value_net")
    graph.add_edge("policy", "replay", source_port="action", target_port="transition")
    
    return graph

if __name__ == "__main__":
    print(ppo_graph())
