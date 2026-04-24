"""
Graph Optimizer for performing dead node elimination, constant folding, and node fusion.
"""

from core.graph import (
    Graph,
    Node,
    Edge,
    NodeId,
    NODE_TYPE_SINK,
    NODE_TYPE_METRICS_SINK,
    NODE_TYPE_TARGET_SYNC,
)
from runtime.specs import get_spec
from compiler.rewrite import FusionRule, RewriteEngine
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field
import copy


@dataclass
class OptimizationStep:
    """Represents a single optimization action taken by the compiler."""
    rule_name: str
    pattern: List[str]
    replacement: str
    removed_nodes: List[NodeId]
    new_node: NodeId

class OptimizationReport:
    """Collection of optimization steps applied to a graph."""
    def __init__(self):
        self.steps: List[OptimizationStep] = []
        self.dead_nodes_removed: List[NodeId] = []
        self.skipped_fusions: List[Dict[str, Any]] = []

    def add_step(self, step: OptimizationStep):
        self.steps.append(step)

    def add_dead_node(self, node_id: NodeId):
        self.dead_nodes_removed.append(node_id)

    def add_skipped_fusion(self, rule_name: str, nodes: List[NodeId], reason: str):
        self.skipped_fusions.append({
            "rule": rule_name,
            "nodes": nodes,
            "reason": reason
        })

    def __str__(self) -> str:
        lines = []
        if self.dead_nodes_removed:
            lines.append(f"Dead Node Elimination: Removed {len(self.dead_nodes_removed)} nodes")
            lines.append(f"  Nodes: {', '.join(map(str, self.dead_nodes_removed))}")
        
        for step in self.steps:
            pattern_str = " -> ".join(step.pattern)
            lines.append(f"Applied rule {step.rule_name}:")
            lines.append(f"  [{pattern_str}] => [{step.replacement}]")
            lines.append(f"  Removed nodes: {', '.join(map(str, step.removed_nodes))}")

        for skipped in self.skipped_fusions:
            lines.append(f"Skipped rule {skipped['rule']} for nodes {skipped['nodes']}:")
            lines.append(f"  Reason: {skipped['reason']}")

        return "\n".join(lines)

# Global Rewrite Engine with default rules
OPTIMIZER_ENGINE = RewriteEngine()

# Register default fusion rules
OPTIMIZER_ENGINE.add_rule(
    FusionRule(
        name="greedy_policy",
        pattern=["QValuesSingle", "Argmax"],
        replacement="GreedyPolicy",
    )
)


def dead_node_elimination(graph: Graph, report: Optional[OptimizationReport] = None) -> Graph:
    """
    Removes nodes that do not contribute to any Sink node and have no side effects.
    """
    # 0. Guard: If the graph is fundamentally malformed (dangling edges),
    # skip optimization and let validate_structure handle the errors.
    for edge in graph.edges:
        if edge.src not in graph.nodes or edge.dst not in graph.nodes:
            return graph

    # 1. Identify "initially live" nodes: Sinks and side-effect nodes
    live_nodes: Set[NodeId] = set()
    for nid, node in graph.nodes.items():
        spec = get_spec(node.node_type)
        is_sink = node.node_type in [
            NODE_TYPE_SINK,
            NODE_TYPE_METRICS_SINK,
            NODE_TYPE_TARGET_SYNC,
        ]
        has_side_effects = spec and (len(spec.side_effects) > 0 or not spec.pure)

        if is_sink or has_side_effects:
            live_nodes.add(nid)

    # 2. Propagate liveness backwards
    changed = True
    while changed:
        changed = False
        for edge in graph.edges:
            if edge.dst in live_nodes and edge.src not in live_nodes:
                live_nodes.add(edge.src)
                changed = True

    # 3. Create new graph with only live nodes
    removed_nodes = [nid for nid in graph.nodes if nid not in live_nodes]
    if removed_nodes:
        print(f"[DNE] Removing dead nodes: {removed_nodes}")
    new_graph = Graph()
    new_graph.nodes = {
        nid: node for nid, node in graph.nodes.items() if nid in live_nodes
    }
    if report:
        for nid in graph.nodes:
            if nid not in live_nodes:
                report.add_dead_node(nid)

    new_graph.edges = [
        edge
        for edge in graph.edges
        if edge.src in live_nodes and edge.dst in live_nodes
    ]

    # Rebuild adjacency
    for edge in new_graph.edges:
        if edge.src not in new_graph._adjacency:
            new_graph._adjacency[edge.src] = set()
        new_graph._adjacency[edge.src].add(edge.dst)

    return new_graph


def node_fusion(graph: Graph, report: Optional[OptimizationReport] = None) -> Graph:
    """
    Fuses common patterns into optimized nodes using the RewriteEngine.
    Example: QValuesSingle + Argmax -> GreedyPolicy
    """
    return OPTIMIZER_ENGINE.apply(graph, report=report)


def optimize_graph(graph: Graph, report: Optional[OptimizationReport] = None) -> Graph:
    """Entry point for all graph optimizations."""
    g = dead_node_elimination(graph, report=report)
    g = node_fusion(g, report=report)
    return g
