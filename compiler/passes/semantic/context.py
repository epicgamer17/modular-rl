from core.graph import Graph
from compiler.validation import ValidationReport, ValidationIssue, SEVERITY_ERROR
from runtime.registry import get_spec

def validate_context(graph: Graph) -> ValidationReport:
    """
    Validates that nodes are used in appropriate execution contexts.
    
    Checks:
    - Side-effecting nodes in inference/actor graphs (E401).
    """
    report = ValidationReport()
    
    is_actor_graph = "Actor" in graph.tags or "Inference" in graph.tags
    
    for nid, node in graph.nodes.items():
        spec = get_spec(node.node_type)
        if not spec:
            continue
            
        if is_actor_graph:
            if spec.updates_params or spec.math_category == "optimizer":
                report.add(ValidationIssue(
                    severity=SEVERITY_ERROR,
                    code="E401",
                    node_id=nid,
                    message=f"Side-effect violation: Node '{nid}' ({node.node_type}) is an optimizer or updates parameters, illegal in Actor/Inference graph."
                ))
            if spec.reads_buffer or spec.math_category == "buffer_io":
                report.add(ValidationIssue(
                    severity=SEVERITY_ERROR,
                    code="E401",
                    node_id=nid,
                    message=f"Side-effect violation: Node '{nid}' ({node.node_type}) reads from buffer, illegal in Actor/Inference graph."
                ))
            if spec.math_category == "loss":
                report.add(ValidationIssue(
                    severity=SEVERITY_ERROR,
                    code="E401",
                    node_id=nid,
                    message=f"Semantic violation: Node '{nid}' ({node.node_type}) is a loss node, illegal in Actor/Inference graph."
                ))
    
    return report
