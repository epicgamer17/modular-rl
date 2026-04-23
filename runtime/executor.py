"""
Minimal runtime executor for the RL IR.
Handles topological sorting and sequential execution of graph nodes.
"""

from typing import Dict, Any, List, Set, Callable
from core.graph import Graph, NodeId, Node

# Global operator registry mapping node_type -> execution function
# def run(node: Node, inputs: Dict[NodeId, Any]) -> Any
OPERATOR_REGISTRY: Dict[str, Callable[[Node, Dict[NodeId, Any]], Any]] = {}

def register_operator(node_type: str, func: Callable[[Node, Dict[NodeId, Any]], Any]):
    """Registers an execution function for a node type."""
    OPERATOR_REGISTRY[node_type] = func

def execute(graph: Graph, initial_inputs: Dict[NodeId, Any]) -> Dict[NodeId, Any]:
    """
    Executes the graph using the provided initial inputs for source nodes.
    
    Steps:
    1. Perform topological sort of the graph.
    2. Iterate through nodes in order.
    3. For each node, gather outputs from predecessors as inputs.
    4. Call the registered operator function.
    5. Store the output for downstream nodes.
    
    Returns:
        A dictionary mapping NodeId to their computed outputs.
    """
    # 1. Topological Sort (Kahn's Algorithm)
    order = _topological_sort(graph)
    
    # 2. Execution
    node_outputs: Dict[NodeId, Any] = {}
    
    # Seed initial inputs (for source nodes)
    for nid, val in initial_inputs.items():
        node_outputs[nid] = val
        
    for nid in order:
        if nid in node_outputs and nid in initial_inputs:
            # Source node already materialized
            continue
            
        node = graph.nodes[nid]
        
        # Gather inputs from predecessors
        # For now, we pass a dict of {pred_id: output}
        inputs = {}
        for edge in graph.edges:
            if edge.dst == nid:
                if edge.src not in node_outputs:
                    raise RuntimeError(f"Input {edge.src} for node {nid} not yet computed.")
                inputs[edge.src] = node_outputs[edge.src]
        
        # Execute operator
        if node.node_type not in OPERATOR_REGISTRY:
            raise RuntimeError(f"No operator registered for node type: {node.node_type}")
            
        op_func = OPERATOR_REGISTRY[node.node_type]
        output = op_func(node, inputs)
        node_outputs[nid] = output
        
    return node_outputs

def _topological_sort(graph: Graph) -> List[NodeId]:
    """Returns a list of NodeIds in topological order."""
    # Build in-degree map
    in_degree = {nid: 0 for nid in graph.nodes}
    for edges in graph.adjacency_list.values():
        for dst in edges:
            in_degree[dst] += 1
            
    # Queue for nodes with 0 in-degree
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    order = []
    
    while queue:
        u = queue.pop(0)
        order.append(u)
        
        for v in graph.adjacency_list.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(order) != len(graph.nodes):
        raise ValueError("Graph contains cycles; cannot perform topological sort.")
        
    return order
