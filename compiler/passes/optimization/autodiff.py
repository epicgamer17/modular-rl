"""
Autodiff lowering pass for RL IR.
Inserts explicit Backward nodes downstream of Loss nodes.
"""

from typing import List, Dict, Set, Optional, Any
from core.graph import Graph, EdgeType
from runtime.registry import get_spec


def autodiff(graph: Graph, report: Optional[Any] = None) -> Graph:
    """
    Transforms the graph by inserting explicit Backward nodes.

    Logic:
    1. Find all nodes that creates_grad (Loss nodes).
    2. For each Loss node, identify the parameter handles it affects.
    3. Insert a Backward node downstream of the Loss node.
    4. Connect the Backward node to the Optimizer if one exists.
    """
    # 1. Identify loss nodes
    loss_nodes = []
    for nid, node in graph.nodes.items():
        spec = get_spec(node.node_type)
        if spec and spec.creates_grad:
            loss_nodes.append(nid)

    if not loss_nodes:
        return graph

    # 2. For each loss node, find which optimizers consume it
    # and which parameters it affects.
    for loss_id in loss_nodes:
        # Find downstream optimizers
        optimizers = []
        for edge in graph.edges:
            if edge.src == loss_id and edge.edge_type == EdgeType.DATA:
                dst_node = graph.nodes[edge.dst]
                dst_spec = get_spec(dst_node.node_type)
                if dst_spec and dst_spec.consumes_grad:
                    optimizers.append(edge.dst)

        if not optimizers:
            # If no explicit optimizer, we still add a Backward node if it's a leaf
            # or if requested. For now, only insert if there's an optimizer to update.
            continue

        for opt_id in optimizers:
            opt_node = graph.nodes[opt_id]
            model_handle = opt_node.params.get("model_handle")

            # Create Backward node
            backward_id = f"backward_{loss_id}"
            if backward_id not in graph.nodes:
                graph.add_node(
                    backward_id, "Backward", params={"model_handle": model_handle}
                )

                # Connect Loss -> Backward
                graph.add_edge(loss_id, backward_id, dst_port="loss")
                
                if report:
                    report.add_backward_pass(loss_id, model_handle)

            # Let's add GradBuffer nodes as well
            grad_id = f"grads_{model_handle}"
            if grad_id not in graph.nodes:
                graph.add_node(
                    grad_id, "GradBuffer", params={"model_handle": model_handle}
                )
                # GradBuffer depends on Backward being done
                graph.add_edge(backward_id, grad_id, edge_type=EdgeType.CONTROL)

    return graph
