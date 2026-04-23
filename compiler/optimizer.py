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
import copy


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


def dead_node_elimination(graph: Graph) -> Graph:
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
    new_graph = Graph()
    new_graph.nodes = {
        nid: node for nid, node in graph.nodes.items() if nid in live_nodes
    }
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


def node_fusion(graph: Graph) -> Graph:
    """
    Fuses common patterns into optimized nodes using the RewriteEngine.
    Example: QValuesSingle + Argmax -> GreedyPolicy
    """
    return OPTIMIZER_ENGINE.apply(graph)


def optimize_graph(graph: Graph) -> Graph:
    """Entry point for all graph optimizations."""
    g = dead_node_elimination(graph)
    g = node_fusion(g)
    return g
