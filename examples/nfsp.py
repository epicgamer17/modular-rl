from rlir.core.graph import GraphDef
from rlir.core.tags import Effect

def nfsp_graph():
    graph = GraphDef("NFSP_Two_Policy")
    
    # Nodes
    graph.add_source("game", "LeducPoker")
    
    # NFSP uses two policies: Average and Best Response
    graph.add_actor("avg_policy", "SupervisedPolicy")
    graph.add_actor("br_policy", "DQNPolicy")
    
    # Control node selects which policy to use
    graph.add_control("policy_selector", "AnticipatoryParameter", 
                      inputs=["avg", "br"], 
                      outputs=["selected_action"])
    
    graph.add_sink("sl_buffer", "ReservoirBuffer")
    graph.add_sink("rl_buffer", "ReplayBuffer")
    
    # Edges
    graph.add_edge("game", "avg_policy")
    graph.add_edge("game", "br_policy")
    graph.add_edge("avg_policy", "policy_selector", source_port="action", target_port="avg")
    graph.add_edge("br_policy", "policy_selector", source_port="action", target_port="br")
    
    return graph

if __name__ == "__main__":
    print(nfsp_graph())
