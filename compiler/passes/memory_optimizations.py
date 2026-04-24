"""
Memory optimization passes for training graphs.
"""

from typing import Dict, List, Optional, Set, Tuple, Any

from core.graph import Graph, Node, NodeId
from runtime.specs import get_spec


def _clone_with_params(node: Node, params: Dict[str, object]) -> Node:
    return Node(
        node_id=node.node_id,
        node_type=node.node_type,
        schema_in=node.schema_in,
        schema_out=node.schema_out,
        params=params,
        tags=node.tags,
    )


def _copy_graph(graph: Graph) -> Graph:
    new_graph = Graph()
    new_graph.nodes = dict(graph.nodes)
    new_graph.edges = list(graph.edges)
    new_graph._adjacency = {nid: set(dsts) for nid, dsts in graph._adjacency.items()}
    new_graph.parameters = dict(graph.parameters)
    return new_graph


def _topological_order(graph: Graph) -> List[NodeId]:
    in_degree = {nid: 0 for nid in graph.nodes}
    for edge in graph.edges:
        if edge.dst in in_degree:
            in_degree[edge.dst] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    order: List[NodeId] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for succ in graph.adjacency_list.get(nid, set()):
            if succ in in_degree:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
    return order


def apply_activation_checkpointing(graph: Graph) -> Graph:
    """
    Marks trainable forward/loss nodes so runtime operators can checkpoint them.
    """
    new_graph = _copy_graph(graph)
    for nid, node in graph.nodes.items():
        spec = get_spec(node.node_type)
        if spec is None:
            continue

        should_checkpoint = False
        if spec.creates_grad and "model_handle" in node.params:
            should_checkpoint = True
        elif (
            spec.differentiable
            and not spec.creates_grad
            and not spec.consumes_grad
            and not spec.updates_params
            and "model_handle" in node.params
            and "target" not in str(node.params.get("model_handle", "")).lower()
        ):
            should_checkpoint = True

        if should_checkpoint:
            params = dict(node.params)
            params["activation_checkpoint"] = True
            new_graph.nodes[nid] = _clone_with_params(node, params)

    return new_graph


def hoist_no_grad_regions(graph: Graph, report: Optional[Any] = None) -> Graph:
    """
    Marks target-network / inference-only regions so operators evaluate them under no-grad.
    """
    new_graph = _copy_graph(graph)
    for nid, node in graph.nodes.items():
        spec = get_spec(node.node_type)
        if spec is None:
            continue

        should_hoist = False
        model_handle = str(node.params.get("model_handle", ""))
        target_handle = str(node.params.get("target_handle", ""))
        
        branch_name = None
        if "target" in model_handle.lower():
            should_hoist = True
            branch_name = "target_q" if "q" in model_handle.lower() else model_handle
        elif "target" in target_handle.lower():
            should_hoist = True
            branch_name = "target_q" if "q" in target_handle.lower() else target_handle
        elif node.node_type in {"BellmanTarget"}:
            should_hoist = True
            branch_name = "target_q"

        if should_hoist:
            params = dict(node.params)
            params["no_grad_region"] = True
            new_graph.nodes[nid] = _clone_with_params(node, params)
            
            if report and branch_name:
                report.add_hoisted_no_grad(branch_name)

    return new_graph


def assign_shared_buffer_reuse(graph: Graph) -> Graph:
    """
    Assigns reusable buffer slots to temporary nodes with non-overlapping live ranges.
    """
    order = _topological_order(graph)
    index_of = {nid: idx for idx, nid in enumerate(order)}

    candidates: List[Tuple[NodeId, int, int]] = []
    for nid in order:
        node = graph.nodes[nid]
        spec = get_spec(node.node_type)
        if spec is None or not spec.pure or spec.stateful or spec.side_effects:
            continue
        has_outputs = bool(node.schema_out.fields) or bool(spec.outputs)
        if not has_outputs:
            continue

        start = index_of[nid]
        users = [edge.dst for edge in graph.edges if edge.src == nid and edge.dst in index_of]
        end = max((index_of[user] for user in users), default=start)
        candidates.append((nid, start, end))

    active_slots: List[Tuple[int, int]] = []  # (slot, live_until)
    next_slot = 0
    assigned: Dict[NodeId, int] = {}

    for nid, start, end in candidates:
        reusable_slot: Optional[int] = None
        still_active: List[Tuple[int, int]] = []
        for slot, live_until in active_slots:
            if live_until < start and reusable_slot is None:
                reusable_slot = slot
            else:
                still_active.append((slot, live_until))
        active_slots = still_active

        if reusable_slot is None:
            reusable_slot = next_slot
            next_slot += 1

        assigned[nid] = reusable_slot
        active_slots.append((reusable_slot, end))

    if not assigned:
        return graph

    new_graph = _copy_graph(graph)
    for nid, slot in assigned.items():
        node = graph.nodes[nid]
        params = dict(node.params)
        params["buffer_slot"] = slot
        new_graph.nodes[nid] = _clone_with_params(node, params)

    return new_graph


def optimize_memory(graph: Graph, report: Optional[Any] = None) -> Graph:
    graph = apply_activation_checkpointing(graph)
    graph = hoist_no_grad_regions(graph, report=report)
    graph = assign_shared_buffer_reuse(graph)
    return graph
