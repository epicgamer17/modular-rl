"""
Structural validation pass for the RL IR graph.
Checks for basic connectivity, cycles, and semantic structure.
"""

from collections import deque
from typing import Set, Dict

from core.graph import Graph, NodeId, NODE_TYPE_SOURCE, NODE_TYPE_SINK, NODE_TYPE_REPLAY_QUERY
from runtime.registry import get_spec
from compiler.validation import (
    ValidationIssue,
    ValidationReport,
    SEVERITY_ERROR,
    SEVERITY_WARN,
)


def validate_structure(graph: Graph) -> ValidationReport:
    """
    Performs structural validation on the provided graph.

    Checks for:
    1. Edge references missing nodes (E001).
    2. Cycles via topological sort (E002).
    3. Unreachable nodes from any SOURCE (W001).
    4. Sinkless branches that start from SOURCE but never reach a SINK (E003).

    Args:
        graph: The Graph instance to validate.

    Returns:
        A ValidationReport containing any discovered issues.
    """
    report = ValidationReport()

    # 1. Edge references missing node
    # We check all edges against the current nodes in the graph.
    for edge in graph.edges:
        if edge.src not in graph.nodes:
            report.add(
                ValidationIssue(
                    severity=SEVERITY_ERROR,
                    code="E001",
                    node_id=str(edge.src),
                    message=f"Edge source '{edge.src}' missing from nodes",
                )
            )
        if edge.dst not in graph.nodes:
            report.add(
                ValidationIssue(
                    severity=SEVERITY_ERROR,
                    code="E001",
                    node_id=str(edge.dst),
                    message=f"Edge destination '{edge.dst}' missing from nodes",
                )
            )

    # 2. Cycle Detection (using Kahn's algorithm for topological sort)
    # This also serves to verify that the graph is a DAG.
    in_degree: Dict[NodeId, int] = {nid: 0 for nid in graph.nodes}
    for edge in graph.edges:
        # Only count edges between existing nodes to avoid double-reporting E001
        if edge.src in graph.nodes and edge.dst in graph.nodes:
            in_degree[edge.dst] += 1

    queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
    visited_count = 0
    while queue:
        u = queue.popleft()
        visited_count += 1
        # Use graph._adjacency directly or adjacency_list property
        for v in graph.adjacency_list.get(u, set()):
            if v in in_degree:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

    if visited_count < len(graph.nodes):
        report.add(
            ValidationIssue(
                severity=SEVERITY_ERROR,
                code="E002",
                node_id=None,
                message="Graph contains cycles (not a Directed Acyclic Graph)",
            )
        )

    # 3. Reachability from SOURCE
    # Nodes that cannot be reached from any node of type SOURCE or REPLAY_QUERY.
    reachable: Set[NodeId] = set()
    sources = [
        nid for nid, node in graph.nodes.items() 
        if node.node_type in [NODE_TYPE_SOURCE, NODE_TYPE_REPLAY_QUERY]
    ]

    stack = list(sources)
    while stack:
        u = stack.pop()
        if u not in reachable:
            reachable.add(u)
            for v in graph.adjacency_list.get(u, set()):
                if v in graph.nodes:  # Only traverse to existing nodes
                    stack.append(v)

    for nid in graph.nodes:
        if nid not in reachable:
            report.add(
                ValidationIssue(
                    severity=SEVERITY_WARN,
                    code="W001",
                    node_id=str(nid),
                    message=f"Node '{nid}' is unreachable from any SOURCE",
                )
            )

    # 4. Sinkless source branches
    # Every node reachable from a SOURCE should eventually reach a node of type SINK
    # or an impure node (which performs side effects like logging or state updates).
    has_path_to_sink: Set[NodeId] = set()
    sinks = []
    for nid, node in graph.nodes.items():
        spec = get_spec(node.node_type)
        is_explicit_sink = node.node_type == NODE_TYPE_SINK
        has_side_effects = spec and (not spec.pure or len(spec.side_effects) > 0)
        
        if is_explicit_sink or has_side_effects:
            sinks.append(nid)

    # Build reverse adjacency for backward traversal from sinks
    reverse_adj: Dict[NodeId, Set[NodeId]] = {nid: set() for nid in graph.nodes}
    for edge in graph.edges:
        if edge.src in graph.nodes and edge.dst in graph.nodes:
            reverse_adj[edge.dst].add(edge.src)

    stack = list(sinks)
    while stack:
        u = stack.pop()
        if u not in has_path_to_sink:
            has_path_to_sink.add(u)
            for v in reverse_adj.get(u, set()):
                stack.append(v)

    for nid in reachable:
        node = graph.nodes[nid]
        # A node is valid if it's a sink itself or can reach one
        is_sink = nid in sinks
        if not is_sink and nid not in has_path_to_sink:
            report.add(
                ValidationIssue(
                    severity=SEVERITY_ERROR,
                    code="E003",
                    node_id=str(nid),
                    message=f"Node '{nid}' belongs to a sinkless branch (cannot reach any SINK)",
                )
            )

    return report
