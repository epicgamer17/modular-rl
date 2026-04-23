"""
Graph validation logic for the RL IR.
Ensures structural, type-level, and semantic correctness of graphs.
"""

from typing import List, Dict, Set, Optional
from core.graph import Graph, EdgeType, NodeId
from core.schema import TAG_ON_POLICY, TAG_ORDERED, TAG_OFF_POLICY

def validate_graph(graph: Graph) -> None:
    """
    Performs full validation on a Graph instance.
    
    Checks:
    - Structural: No cycles (must be a DAG).
    - Connectivity: All nodes must be reachable from at least one source (optional, but good).
    - Type Compatibility: Output schemas must match input schemas for data edges.
    - Semantic Constraints: Rule-based checks for specific node types/tags.
    
    Raises:
        ValueError: If any validation rule is violated.
    """
    # 1. Structural Checks
    _check_cycles(graph)
    _check_connectivity(graph)
    
    # 2. Type Compatibility
    _check_type_compatibility(graph)
    
    # 3. Semantic Constraints
    _check_semantic_constraints(graph)

def _check_cycles(graph: Graph) -> None:
    """Ensures the graph is a Directed Acyclic Graph (DAG)."""
    visited = set()
    path = set()
    
    def visit(nid: NodeId):
        if nid in path:
            raise ValueError(f"Cycle detected in graph at node: {nid}")
        if nid in visited:
            return
        
        path.add(nid)
        for neighbor in graph.adjacency_list.get(nid, []):
            visit(neighbor)
        path.remove(nid)
        visited.add(nid)

    for nid in graph.nodes:
        visit(nid)

def _check_connectivity(graph: Graph) -> None:
    """Basic connectivity check. For now, ensure no completely isolated nodes."""
    if not graph.nodes:
        return
        
    adj = graph.adjacency_list
    # Nodes with incoming edges
    has_incoming = set()
    for edges in adj.values():
        for dst in edges:
            has_incoming.add(dst)
            
    # Nodes with outgoing edges
    has_outgoing = set()
    for src, edges in adj.items():
        if edges:
            has_outgoing.add(src)
            
    for nid in graph.nodes:
        if nid not in has_incoming and nid not in has_outgoing and len(graph.nodes) > 1:
            raise ValueError(f"Isolated node detected: {nid}")

def _check_type_compatibility(graph: Graph) -> None:
    """Verifies that data flows between nodes have compatible schemas."""
    for edge in graph.edges:
        if edge.edge_type == EdgeType.DATA:
            src_node = graph.nodes[edge.src]
            dst_node = graph.nodes[edge.dst]
            
            if not src_node.schema_out.is_compatible(dst_node.schema_in):
                raise ValueError(
                    f"Type mismatch on edge {edge.src} -> {edge.dst}: "
                    f"Output schema {src_node.schema_out} is not compatible with "
                    f"Input schema {dst_node.schema_in}"
                )

def _check_semantic_constraints(graph: Graph) -> None:
    """Enforces semantic rules based on node tags."""
    for nid, node in graph.nodes.items():
        # PPO Rule: PPO nodes must be marked as OnPolicy
        if "PPO" in node.tags:
            if TAG_ON_POLICY not in node.tags:
                raise ValueError(f"Semantic error: PPO node {nid} must have {TAG_ON_POLICY} tag.")
            
            # Check for Replay buffer in predecessors
            _check_no_replay_predecessor(graph, nid)

        # GAE Rule: GAE nodes require Ordered input trajectories
        if "GAE" in node.tags:
            is_ordered = any(TAG_ORDERED in spec.tags for spec in node.schema_in.get_field_map().values())
            if not is_ordered:
                raise ValueError(f"Semantic error: GAE node {nid} requires {TAG_ORDERED} input trajectory.")

def _check_no_replay_predecessor(graph: Graph, target_nid: NodeId) -> None:
    """Helper to ensure a node doesn't have a Replay buffer in its upstream path."""
    # Build reverse adjacency for upstream traversal
    reverse_adj: Dict[NodeId, List[NodeId]] = {nid: [] for nid in graph.nodes}
    for src, dsts in graph.adjacency_list.items():
        for dst in dsts:
            reverse_adj[dst].append(src)
            
    visited = set()
    stack = [target_nid]
    
    while stack:
        nid = stack.pop()
        if nid in visited:
            continue
        visited.add(nid)
        
        node = graph.nodes[nid]
        if "Replay" in node.tags:
            raise ValueError(f"Semantic error: OnPolicy node {target_nid} cannot consume data from Replay buffer {nid}.")
            
        stack.extend(reverse_adj.get(nid, []))
