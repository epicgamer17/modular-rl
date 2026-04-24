"""
Gradient Flow Analyzer for RL IR graphs.
Detects paths from Parameters to Optimizer through Forward, Loss, and Backward passes.
Identifies dead parameters, detached tensors, and unused branches.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from core.graph import Graph, NodeId, EdgeType
from runtime.specs import get_spec


@dataclass
class GradientReport:
    """Report containing gradient flow analysis results."""

    params_with_grad: List[str] = field(default_factory=list)
    params_without_grad: List[str] = field(default_factory=list)
    detached_edges: List[Tuple[NodeId, NodeId]] = field(default_factory=list)
    unused_branches: List[NodeId] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def analyze_gradients(graph: Graph) -> GradientReport:
    """
    Performs gradient flow analysis on the graph.
    
    Flow: Parameter -> Forward (differentiable) -> Loss (creates_grad) -> Optimizer (consumes_grad).
    """
    report = GradientReport()

    # 1. Identify key nodes and handles
    loss_nodes = []
    optimizer_nodes = []
    param_nodes = {}  # handle -> list of node IDs using it

    for nid, node in graph.nodes.items():
        spec = get_spec(node.node_type)
        if not spec:
            continue

        if spec.creates_grad:
            loss_nodes.append(nid)
        if spec.consumes_grad:
            optimizer_nodes.append(nid)

        if spec.parameter_handles:
            for handle_key in spec.parameter_handles:
                if "opt" in handle_key.lower():
                    continue
                handle_val = node.params.get(handle_key)
                if handle_val:
                    if handle_val not in param_nodes:
                        param_nodes[handle_val] = []
                    param_nodes[handle_val].append(nid)

    # 2. Build adjacency lists (Robust to dangling edges)
    backward_adj = {nid: [] for nid in graph.nodes}
    forward_adj = {nid: [] for nid in graph.nodes}
    for edge in graph.edges:
        if edge.edge_type == EdgeType.DATA:
            if edge.dst in backward_adj:
                backward_adj[edge.dst].append(edge.src)
            if edge.src in forward_adj:
                forward_adj[edge.src].append(edge.dst)

    # 3. Identify losses that reach an optimizer
    # (In our IR, Optimizer nodes consume Loss nodes)
    active_losses = set()
    for opt_id in optimizer_nodes:
        for prev in backward_adj.get(opt_id, []):
            if prev in loss_nodes:
                active_losses.add(prev)

    # 4. Backward reachability from active losses through differentiable paths
    diff_reachable = set()

    def traverse_backward_diff(start_nodes):
        stack = list(start_nodes)
        visited = set(start_nodes)
        while stack:
            curr = stack.pop()
            # If the current node is non-differentiable, it stops the gradient flow
            spec = get_spec(graph.nodes[curr].node_type)
            if spec and not spec.differentiable:
                continue

            for prev in backward_adj.get(curr, []):
                if prev not in visited:
                    visited.add(prev)
                    stack.append(prev)
        return visited

    diff_reachable = traverse_backward_diff(active_losses)

    # 5. Analyze Parameters
    for handle, nodes in param_nodes.items():
        # A handle is "alive" if it has a differentiable path to an active optimizer
        is_trainable = False
        for nid in nodes:
            if nid in diff_reachable:
                is_trainable = True
                break

        if is_trainable:
            report.params_with_grad.append(handle)
        else:
            report.params_without_grad.append(handle)
            
            # Check for ANY path to ANY loss (even non-differentiable)
            has_path_to_loss = False
            for nid in nodes:
                # Simple BFS/DFS to find ANY path to ANY loss_node
                q = [nid]
                visited = {nid}
                while q:
                    curr = q.pop(0)
                    if curr in loss_nodes:
                        has_path_to_loss = True
                        break
                    for succ in forward_adj.get(curr, []):
                        if succ not in visited:
                            visited.add(succ)
                            q.append(succ)
                if has_path_to_loss:
                    break
            
            if not has_path_to_loss:
                report.warnings.append(
                    f"Dead parameter detected: '{handle}'. Model exists but no path to any loss node."
                )

    # 6. Detect Detached Edges
    # Edge U -> V where U leads to V, V can reach a loss, but V (or the path) is non-differentiable.
    
    # Forward reachability from params
    param_source_nodes = []
    for nodes in param_nodes.values():
        param_source_nodes.extend(nodes)
        
    forward_from_params = set(param_source_nodes)
    stack = list(param_source_nodes)
    while stack:
        curr = stack.pop()
        for succ in forward_adj.get(curr, []):
            if succ not in forward_from_params:
                forward_from_params.add(succ)
                stack.append(succ)

    # Backward reachability from losses (any loss, not just active)
    reachable_to_any_loss = set(loss_nodes)
    stack = list(loss_nodes)
    while stack:
        curr = stack.pop()
        for prev in backward_adj.get(curr, []):
            if prev not in reachable_to_any_loss:
                reachable_to_any_loss.add(prev)
                stack.append(prev)

    for edge in graph.edges:
        if edge.src in forward_from_params and edge.dst in reachable_to_any_loss:
            spec = get_spec(graph.nodes[edge.dst].node_type)
            if spec and not spec.differentiable:
                report.detached_edges.append((edge.src, edge.dst))
                report.warnings.append(
                    f"Detached gradient flow at edge {edge.src} -> {edge.dst}. Operator '{spec.name}' is non-differentiable."
                )

    # 7. Unused Branches
    for nid, node in graph.nodes.items():
        if nid in forward_from_params and nid not in diff_reachable:
            # If it's a differentiable leaf node that isn't a sink/optimizer
            if not forward_adj.get(nid):
                spec = get_spec(node.node_type)
                if spec and spec.differentiable:
                    # Ignore standard Sinks
                    if "Sink" not in node.node_type:
                        report.unused_branches.append(nid)
                        report.warnings.append(
                            f"Unused differentiable branch: '{nid}'. Produces values that are not consumed by any active gradient path."
                        )

    return report
