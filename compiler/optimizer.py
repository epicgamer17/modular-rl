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
from runtime.registry import get_spec
from compiler.rewrite import FusionRule, RewriteEngine
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field
import copy
from compiler.passes.optimization.memory import optimize_memory


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
        self.trainable_params: List[str] = []
        self.backward_passes: List[Dict[str, str]] = []
        self.hoisted_no_grad: List[str] = []

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

    def add_trainable_param(self, param_name: str):
        if param_name not in self.trainable_params:
            self.trainable_params.append(param_name)

    def add_backward_pass(self, loss_node: NodeId, backward_node: NodeId):
        self.backward_passes.append({
            "loss": str(loss_node),
            "backward": str(backward_node)
        })

    def add_hoisted_no_grad(self, branch_name: str):
        if branch_name not in self.hoisted_no_grad:
            self.hoisted_no_grad.append(branch_name)

    def __str__(self) -> str:
        lines = []
        
        if self.trainable_params:
            lines.append("Detected trainable params:")
            for p in self.trainable_params:
                lines.append(f"  {p}")
            lines.append("")

        if self.backward_passes:
            lines.append("Inserted backward pass:")
            for bp in self.backward_passes:
                lines.append(f"  {bp['loss']} -> Backward({bp['backward']})")
            lines.append("")

        if self.dead_nodes_removed:
            lines.append(f"Dead Node Elimination: Removed {len(self.dead_nodes_removed)} nodes")
            lines.append(f"  Nodes: {', '.join(map(str, self.dead_nodes_removed))}")
            lines.append("")
        
        for step in self.steps:
            pattern_str = " -> ".join(step.pattern)
            lines.append(f"Applied rule {step.rule_name}:")
            lines.append(f"  [{pattern_str}] => [{step.replacement}]")
            lines.append(f"  Removed nodes: {', '.join(map(str, step.removed_nodes))}")
            lines.append("")

        if self.skipped_fusions:
            lines.append("Skipped fusion:")
            for skipped in self.skipped_fusions:
                lines.append(f"  {skipped['reason']}")
            lines.append("")

        if self.hoisted_no_grad:
            lines.append("Applied no_grad hoist:")
            for branch in self.hoisted_no_grad:
                lines.append(f"  {branch} branch")
            lines.append("")

        return "\n".join(lines).strip()

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
        from observability.tracing.event_schema import get_emitter, EventType, Event
        emitter = get_emitter()
        for nid in removed_nodes:
            emitter.emit(Event(
                type=EventType.PRUNING_EVENT,
                name="dead_node_elimination",
                metadata={"node_id": str(nid), "node_type": graph.nodes[nid].node_type}
            ))
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
    g = optimize_memory(g, report=report)
    return g
