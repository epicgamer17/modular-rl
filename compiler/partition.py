from typing import Dict, List, Set, Tuple, Optional, Any
from core.graph import Graph, NodeId, Edge, EdgeType
from runtime.specs import get_spec

def partition_graph(global_graph: Graph) -> Dict[str, Graph]:
    """
    Partitions a global graph into context-specific subgraphs (actor, learner).
    
    Rules:
    1. Nodes are assigned to partitions based on their 'allowed_contexts' spec.
    2. Edges within a partition are preserved.
    3. Edges crossing partitions are replaced by ChannelOut/ChannelIn pairs.
    """
    actor_graph = Graph()
    learner_graph = Graph()
    
    # 1. Assign nodes to partitions
    node_to_partition: Dict[NodeId, str] = {}
    for nid, node in global_graph.nodes.items():
        spec = get_spec(node.node_type)
        allowed = spec.allowed_contexts if spec else {"actor", "learner"}
        
        # Priority: explicit single-context nodes
        if "actor" in allowed and "learner" not in allowed:
            partition = "actor"
        elif "learner" in allowed and "actor" not in allowed:
            partition = "learner"
        else:
            # Shared nodes (Source, Sink, etc.)
            # Heuristic: if it's connected to a learner-only node, it might be learner.
            # For now, let's default based on common RL patterns:
            # Sources are usually in both, but we partition them into both if needed.
            # Here we just pick one.
            partition = "actor" # Default
            
        node_to_partition[nid] = partition

    # 2. Add nodes to their subgraphs
    for nid, node in global_graph.nodes.items():
        partition = node_to_partition[nid]
        target_g = actor_graph if partition == "actor" else learner_graph
        target_g.add_node(
            nid, 
            node.node_type, 
            schema_in=node.schema_in, 
            schema_out=node.schema_out, 
            params=node.params, 
            tags=node.tags
        )

    # 3. Process edges and create channels
    for edge in global_graph.edges:
        src_part = node_to_partition[edge.src]
        dst_part = node_to_partition[edge.dst]
        
        if src_part == dst_part:
            # Internal edge
            target_g = actor_graph if src_part == "actor" else learner_graph
            target_g.add_edge(edge.src, edge.dst, edge_type=edge.edge_type, dst_port=edge.dst_port)
        else:
            # Crossing edge!
            # Create ChannelOut in src partition
            src_g = actor_graph if src_part == "actor" else learner_graph
            channel_id = f"{edge.src}_{edge.dst}_channel"
            
            ch_out_id = f"{channel_id}_out"
            if ch_out_id not in src_g.nodes:
                src_g.add_node(ch_out_id, "ChannelOut", params={"channel_name": channel_id})
            src_g.add_edge(edge.src, ch_out_id, edge_type=edge.edge_type, dst_port=edge.dst_port)
            
            # Create ChannelIn in dst partition
            dst_g = actor_graph if dst_part == "actor" else learner_graph
            ch_in_id = f"{channel_id}_in"
            if ch_in_id not in dst_g.nodes:
                dst_g.add_node(ch_in_id, "ChannelIn", params={"channel_name": channel_id})
            dst_g.add_edge(ch_in_id, edge.dst, edge_type=edge.edge_type, dst_port=edge.dst_port)

    return {"actor": actor_graph, "learner": learner_graph}
