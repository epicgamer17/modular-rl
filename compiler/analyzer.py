"""
Graph Analyzer for performing static analysis on RL IR graphs.
Detects structural inefficiencies and semantic violations before runtime.
"""

from typing import List, Dict, Set, Any
from core.graph import Graph, NODE_TYPE_SOURCE, NODE_TYPE_ACTOR, NODE_TYPE_SINK
from core.schema import TAG_ON_POLICY, TAG_OFF_POLICY

class GraphAnalysis:
    """Container for graph analysis results."""
    def __init__(self):
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def is_valid(self) -> bool:
        return len(self.errors) == 0

def analyze_graph(graph: Graph) -> GraphAnalysis:
    """
    Performs static analysis on the graph:
    1. Unused Node Detection (Dead Code)
    2. Missing Sink Detection
    3. RL Semantic Violation Detection
    """
    analysis = GraphAnalysis()
    
    # 1. Structural Analysis: Dead Code (Unused Nodes)
    # A node is unused if it's not a SINK and has no outgoing edges
    nodes_with_successors: Set[str] = {edge.src for edge in graph.edges}
    for nid, node in graph.nodes.items():
        if node.node_type == NODE_TYPE_SINK:
            continue
        if nid not in nodes_with_successors:
            # Source nodes with no successors are definitely unused
            if node.node_type == NODE_TYPE_SOURCE:
                analysis.warnings.append(f"Unused Source node: '{nid}'. No outgoing edges.")
            else:
                # Other nodes with no successors are 'leaf' nodes that aren't Sinks
                analysis.warnings.append(f"Dangling node: '{nid}'. Has no successors and is not marked as SINK.")

    # 2. RL Semantics Analysis
    for nid, node in graph.nodes.items():
        # PPO Semantics Check
        if "PPO" in node.tags or "ppo" in node.node_type.lower():
            if TAG_ON_POLICY not in node.tags:
                analysis.errors.append(f"PPO Violation: Node '{nid}' is missing the 'OnPolicy' tag.")
            
            # Check if PPO node is connected to a Sink or used downstream
            if nid not in nodes_with_successors:
                 analysis.warnings.append(f"PPO Warning: Node '{nid}' produces a loss but is not connected to an Optimizer or Sink.")

        # Off-Policy / Replay Semantics
        if "Replay" in node.node_type or "DQN" in node.tags:
            if TAG_ON_POLICY in node.tags:
                 analysis.errors.append(f"Semantic Conflict: Node '{nid}' (Off-Policy) is tagged as 'OnPolicy'.")

    # 3. Connectivity Analysis
    nodes_with_predecessors: Set[str] = {edge.dst for edge in graph.edges}
    for nid, node in graph.nodes.items():
        if node.node_type == NODE_TYPE_SOURCE:
            continue
        if nid not in nodes_with_predecessors:
            analysis.errors.append(f"Disconnected node: '{nid}'. Has no incoming edges and is not a SOURCE.")

    return analysis
