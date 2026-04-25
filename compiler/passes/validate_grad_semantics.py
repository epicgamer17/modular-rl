"""
Gradient semantic validation pass.
Enforces safety rules around backward/optimizer structure and context usage.
"""

from typing import Dict, List, Set

from core.graph import EdgeType, Graph
from compiler.validation import (
    SEVERITY_ERROR,
    ValidationIssue,
    ValidationReport,
)
from runtime.registry import get_spec


_EXPLICIT_GRADIENT_NODE_TYPES = {
    "Backward",
    "GradBuffer",
    "AccumulateGrad",
    "OptimizerStepEvery",
}


def _is_gradient_node(node_type: str) -> bool:
    return node_type in _EXPLICIT_GRADIENT_NODE_TYPES


def _build_adjacency(graph: Graph) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    forward_adj: Dict[str, List[str]] = {nid: [] for nid in graph.nodes}
    backward_adj: Dict[str, List[str]] = {nid: [] for nid in graph.nodes}
    for edge in graph.edges:
        if edge.edge_type != EdgeType.DATA:
            continue
        if edge.src in forward_adj and edge.dst in backward_adj:
            forward_adj[edge.src].append(edge.dst)
            backward_adj[edge.dst].append(edge.src)
    return forward_adj, backward_adj


def _build_full_adjacency(graph: Graph) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    forward_adj: Dict[str, List[str]] = {nid: [] for nid in graph.nodes}
    backward_adj: Dict[str, List[str]] = {nid: [] for nid in graph.nodes}
    for edge in graph.edges:
        if edge.src in forward_adj and edge.dst in backward_adj:
            forward_adj[edge.src].append(edge.dst)
            backward_adj[edge.dst].append(edge.src)
    return forward_adj, backward_adj


def _reachable(start: str, adjacency: Dict[str, List[str]], targets: Set[str]) -> bool:
    visited = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        if current in targets and current != start:
            return True
        for nxt in adjacency.get(current, []):
            if nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return False


def validate_grad_semantics(graph: Graph, context: str = "both") -> ValidationReport:
    """
    Checks gradient lifecycle safety.

    Rules:
    - G001: Optimizer with no preceding backward.
    - G002: Backward exists but no optimizer.
    - G003: Same params updated twice in one step.
    - G004: Inference graph accidentally updates params.
    - G005: Actor graph contains gradient nodes.
    """
    report = ValidationReport()
    forward_adj, backward_adj = _build_adjacency(graph)
    full_forward_adj, full_backward_adj = _build_full_adjacency(graph)

    backward_nodes: Set[str] = set()
    optimizer_nodes: Set[str] = set()
    updates_by_handle: Dict[str, List[str]] = {}

    for nid, node in graph.nodes.items():
        spec = get_spec(node.node_type)
        if spec is None:
            continue

        if node.node_type == "Backward":
            backward_nodes.add(nid)
        if spec.consumes_grad:
            optimizer_nodes.add(nid)

        if spec.updates_params:
            for handle_key in spec.parameter_handles or []:
                handle_val = node.params.get(handle_key)
                if not isinstance(handle_val, str):
                    continue
                if "opt" in handle_key.lower():
                    continue
                updates_by_handle.setdefault(handle_val, []).append(nid)

            if context != "learner":
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="G004",
                        node_id=str(nid),
                        message=(
                            f"Node '{nid}' updates parameters in '{context}' context. "
                            "Parameter updates are only allowed in learner graphs."
                        ),
                    )
                )

        if context == "actor" and (
            _is_gradient_node(node.node_type) or spec.creates_grad or spec.consumes_grad
        ):
            report.add(
                ValidationIssue(
                    severity=SEVERITY_ERROR,
                    code="G005",
                    node_id=str(nid),
                    message=(
                        f"Actor graph contains gradient node '{nid}' of type '{node.node_type}'. "
                        "Gradient computation and application must remain in learner graphs."
                    ),
                )
            )

    backward_upstream_losses: Dict[str, Set[str]] = {}
    optimizer_upstream_losses: Dict[str, Set[str]] = {}
    for nid in backward_nodes:
        backward_upstream_losses[nid] = {
            prev for prev in backward_adj.get(nid, []) if prev in graph.nodes
        }
    for nid in optimizer_nodes:
        optimizer_upstream_losses[nid] = {
            prev for prev in backward_adj.get(nid, []) if prev in graph.nodes
        }

    all_losses_with_backward: Set[str] = set()
    for losses in backward_upstream_losses.values():
        all_losses_with_backward.update(losses)

    all_losses_with_optimizer: Set[str] = set()
    for losses in optimizer_upstream_losses.values():
        all_losses_with_optimizer.update(losses)

    for nid in optimizer_nodes:
        loss_inputs = optimizer_upstream_losses.get(nid, set())
        has_loss_association = bool(loss_inputs.intersection(all_losses_with_backward))
        has_backward_control = _reachable(nid, full_backward_adj, backward_nodes)
        if not has_loss_association and not has_backward_control:
            report.add(
                ValidationIssue(
                    severity=SEVERITY_ERROR,
                    code="G001",
                    node_id=str(nid),
                    message=(
                        f"Optimizer node '{nid}' has no associated Backward node for its upstream loss path."
                    ),
                )
            )

    for nid in backward_nodes:
        loss_inputs = backward_upstream_losses.get(nid, set())
        has_loss_association = bool(loss_inputs.intersection(all_losses_with_optimizer))
        has_optimizer_control = _reachable(nid, full_forward_adj, optimizer_nodes)
        if not has_loss_association and not has_optimizer_control:
            report.add(
                ValidationIssue(
                    severity=SEVERITY_ERROR,
                    code="G002",
                    node_id=str(nid),
                    message=(
                        f"Backward node '{nid}' has no associated optimizer for its upstream loss path."
                    ),
                )
            )

    for handle, nodes in updates_by_handle.items():
        if len(nodes) > 1:
            for nid in nodes:
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="G003",
                        node_id=str(nid),
                        message=(
                            f"Parameter handle '{handle}' is updated multiple times in one step "
                            f"by nodes {nodes}."
                        ),
                    )
                )

    return report
