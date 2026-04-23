from rlir.core.graph import GraphDef
from rlir.core.schema import SchemaDef

def dqn_graph():
    graph = GraphDef("DQN_Standard")
    
    # Schemas
    graph.add_schema(SchemaDef("State", {"pixels": "Byte[64,64,3]"}))
    
    # Nodes
    graph.add_source("env", "Atari")
    graph.add_actor("dqn_actor", "EpsilonGreedy")
    graph.add_node("target_q", "QNetwork", metadata={"version": "target"})
    graph.add_sink("replay", "PrioritizedReplay")
    
    # Edges
    graph.add_edge("env", "dqn_actor")
    graph.add_edge("dqn_actor", "replay")
    
    return graph

if __name__ == "__main__":
    print(dqn_graph())
