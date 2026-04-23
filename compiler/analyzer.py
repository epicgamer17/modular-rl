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

    # 3. Connectivity and Type Analysis
    nodes_with_predecessors: Set[str] = {edge.dst for edge in graph.edges}
    for nid, node in graph.nodes.items():
        if node.node_type == NODE_TYPE_SOURCE:
            continue
        if nid not in nodes_with_predecessors:
            analysis.errors.append(f"Disconnected node: '{nid}'. Has no incoming edges and is not a SOURCE.")

    # 4. Type Compatibility Analysis
    for edge in graph.edges:
        src_node = graph.nodes[edge.src]
        dst_node = graph.nodes[edge.dst]
        
        # In a real graph, we'd need to know which output field maps to which input field.
        # For simplicity, we check if schemas are compatible if they are directly connected.
        # This assumes the entire output schema must be compatible with the entire input schema,
        # or we might need a more granular mapping in the Edge object.
        
        if not src_node.schema_out.is_compatible(dst_node.schema_in):
            analysis.errors.append(
                f"Type Mismatch on edge {edge.src} -> {edge.dst}: "
                f"Output schema of {edge.src} is incompatible with input schema of {edge.dst}."
            )

    # 5. Advanced Semantic Type Checks
    for nid, node in graph.nodes.items():
        # Check for logits vs probs misuse
        for field in node.schema_in.fields:
            from core.schema import TensorSpec
            if isinstance(field.spec, TensorSpec) and field.spec.rl_type and "probs" in field.name.lower():
                from core.types import DistributionType
                if isinstance(field.spec.rl_type, DistributionType) and field.spec.rl_type.is_logits:
                    analysis.errors.append(f"Semantic Error: Node '{nid}' expects probabilities for field '{field.name}', but received logits.")

        # Check for stale rollout versions
        for field in node.schema_in.fields:
            from core.schema import TensorSpec
            if isinstance(field.spec, TensorSpec) and field.spec.rl_type:
                from core.types import PolicySnapshotType
                if isinstance(field.spec.rl_type, PolicySnapshotType) and field.spec.rl_type.version < 0:
                     analysis.warnings.append(f"Stale Rollout Warning: Node '{nid}' is using a stale policy snapshot (version {field.spec.rl_type.version}).")

    return analysis
