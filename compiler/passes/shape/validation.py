from core.graph import Graph
from compiler.validation import ValidationReport, ValidationIssue, SEVERITY_ERROR
from runtime.registry import get_spec, is_compatible, format_spec

def validate_shapes(graph: Graph) -> ValidationReport:
    """
    Validates that inferred shapes are consistent with operator expectations.
    
    Checks:
    - Inferred output shape matches static spec if provided (E301).
    - Tensor dimensions are valid for the operator (E302).
    """
    report = ValidationReport()
    
    for nid, node in graph.nodes.items():
        spec = get_spec(node.node_type)
        if not spec:
            continue
            
        # Check output shapes against static specs
        if node.schema_out and node.schema_out.fields:
            out_map = node.schema_out.get_field_map()
            for port_name, port_spec in spec.outputs.items():
                if port_name in out_map:
                    actual = out_map[port_name]
                    expected = port_spec.spec
                    
                    if not is_compatible(actual, expected):
                        report.add(ValidationIssue(
                            severity=SEVERITY_ERROR,
                            code="E301",
                            node_id=nid,
                            message=(
                                f"Inferred shape for port '{port_name}' is incompatible with static spec.\n"
                                f"Expected: {format_spec(expected)}\n"
                                f"Got:      {format_spec(actual)}"
                            )
                        ))
                        
        # 3. Scalar Loss check (Moved from runtime validator)
        is_loss = (
            "loss" in node.node_type.lower()
            or "loss" in node.tags
        )
        if is_loss and node.schema_out and node.schema_out.fields:
            out_map = node.schema_out.get_field_map()
            # If it's a loss node, it should typically have a single scalar output or a port named 'loss'
            loss_port = out_map.get("loss")
            if loss_port:
                if loss_port.shape != () and loss_port.shape != (1,):
                    report.add(ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="E303",
                        node_id=nid,
                        message=f"Loss port must be a scalar, got shape {loss_port.shape}"
                    ))
            elif len(out_map) == 1:
                p_spec = next(iter(out_map.values()))
                if p_spec.shape != () and p_spec.shape != (1,):
                    report.add(ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="E303",
                        node_id=nid,
                        message=f"Loss output must be a scalar, got shape {p_spec.shape}"
                    ))
                        
    return report
