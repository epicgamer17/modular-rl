"""
Metadata validation pass.
Verifies that all nodes have registered metadata specs.
"""

from core.graph import Graph
from compiler.validation import ValidationReport, ValidationIssue, SEVERITY_ERROR, SEVERITY_WARN
from runtime.specs import get_spec


def validate_metadata(graph: Graph, strict: bool = False) -> ValidationReport:
    """
    Checks that all nodes in the graph have registered metadata specs.

    Args:
        graph: The graph to validate.
        strict: If True, missing metadata is an ERROR (mandatory in strict mode).
                Otherwise it's a WARN.
    """
    report = ValidationReport()
    for nid, node in graph.nodes.items():
        spec = get_spec(node.node_type)
        if spec is None:
            # Missing metadata is fatal in strict mode
            severity = SEVERITY_ERROR if strict else SEVERITY_WARN
            report.add(
                ValidationIssue(
                    severity=severity,
                    code="M001",
                    node_id=nid,
                    message=f"Node '{nid}' of type '{node.node_type}' has no registered metadata spec.",
                )
            )
    return report
