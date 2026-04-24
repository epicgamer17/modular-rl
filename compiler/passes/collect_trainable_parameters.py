"""
Compiler pass to collect all trainable parameters referenced in the graph.
"""

from typing import Dict, List, Optional, Any
from core.graph import Graph
from runtime.specs import get_spec


def collect_trainable_parameters(graph: Graph, report: Optional[Any] = None) -> Dict[str, List[str]]:
    """
    Scans the graph for nodes that reference trainable parameters.
    Groups nodes by the parameter handles they reference.

    Args:
        graph: The Graph instance to analyze.
        report: Optional OptimizationReport to record findings.

    Returns:
        A dictionary mapping parameter handles to lists of node IDs that use them.
    """
    param_map: Dict[str, List[str]] = {}

    for nid, node in graph.nodes.items():
        spec = get_spec(node.node_type)
        if not spec or not spec.parameter_handles:
            continue

        for handle_key in spec.parameter_handles:
            if handle_key in node.params:
                handle_val = node.params[handle_key]
                if handle_val not in param_map:
                    param_map[handle_val] = []
                param_map[handle_val].append(str(nid))
                
                if report:
                    report.add_trainable_param(handle_val)

    return param_map
