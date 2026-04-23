"""
Introspection and debugging tools for the RL IR Graph.
Provides utilities to summarize graph structure, trace lineage, and display schemas.
"""

from typing import List, Dict, Set, Optional
from core.graph import Graph, NodeId, EdgeType

def print_graph_summary(graph: Graph) -> None:
    """Prints a human-readable summary of the graph structure."""
    print(f"--- Graph Summary ---")
    print(f"Total Nodes: {len(graph.nodes)}")
    print(f"Total Edges: {len(graph.edges)}")
    print("\nNodes:")
    for nid, node in graph.nodes.items():
        print(f"  - [{node.node_type}] {nid}")
        if node.tags:
            print(f"    Tags: {', '.join(node.tags)}")
        if node.params:
            print(f"    Params: {node.params}")
    print("\nEdges:")
    for edge in graph.edges:
        print(f"  {edge.src} --({edge.edge_type.value})--> {edge.dst}")
    print("-" * 22)

def trace_node_lineage(graph: Graph, target_nid: NodeId) -> Dict[str, Set[NodeId]]:
    """
    Traces the upstream and downstream lineage of a specific node.
    
    Returns:
        A dictionary with 'upstream' and 'downstream' sets of NodeIds.
    """
    if target_nid not in graph.nodes:
        raise ValueError(f"Node {target_nid} not found in graph.")

    upstream = set()
    downstream = set()

    # Build reverse adjacency
    reverse_adj: Dict[NodeId, List[NodeId]] = {nid: [] for nid in graph.nodes}
    for src, dsts in graph.adjacency_list.items():
        for dst in dsts:
            reverse_adj[dst].append(src)

    # Upstream traversal
    stack = [target_nid]
    while stack:
        nid = stack.pop()
        if nid != target_nid:
            upstream.add(nid)
        for parent in reverse_adj.get(nid, []):
            if parent not in upstream:
                stack.append(parent)

    # Downstream traversal
    stack = [target_nid]
    while stack:
        nid = stack.pop()
        if nid != target_nid:
            downstream.add(nid)
        for child in graph.adjacency_list.get(nid, []):
            if child not in downstream:
                stack.append(child)

    return {"upstream": upstream, "downstream": downstream}

def display_schema_propagation(graph: Graph) -> None:
    """Displays how data schemas propagate across graph edges."""
    print(f"--- Schema Propagation ---")
    for edge in graph.edges:
        if edge.edge_type == EdgeType.DATA:
            src_node = graph.nodes[edge.src]
            dst_node = graph.nodes[edge.dst]
            print(f"Edge: {edge.src} -> {edge.dst}")
            print(f"  Output Schema ({edge.src}): {src_node.schema_out}")
            print(f"  Input Schema ({edge.dst}): {dst_node.schema_in}")
            
            # Simple compatibility indicator
            is_compat = src_node.schema_out.is_compatible(dst_node.schema_in)
            print(f"  Compatible: {'YES' if is_compat else 'NO'}")
    print("-" * 26)
