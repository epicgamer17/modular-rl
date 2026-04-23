"""
Port contract validation pass.
Verifies that data flowing between nodes respects their input/output specifications.
"""

from core.graph import Graph, NODE_TYPE_SOURCE, NODE_TYPE_SINK
from core.schema import Schema, TensorSpec
from compiler.validation import (
    ValidationIssue,
    ValidationReport,
    SEVERITY_ERROR,
)
from runtime.specs import get_spec, is_compatible, format_spec


def validate_ports(graph: Graph) -> ValidationReport:
    """
    Validates that all edges in the graph connect compatible ports.

    Checks for:
    - Compatibility between source output and destination input port (E204).
    - Existence of named destination ports (E203).
    - Missing required input ports (E205).

    Args:
        graph: The Graph instance to validate.

    Returns:
        A ValidationReport containing any discovered issues.
    """
    report = ValidationReport()

    # 1. Validate all existing edges
    for edge in graph.edges:
        src_node = graph.nodes.get(edge.src)
        dst_node = graph.nodes.get(edge.dst)

        # Skip if nodes are missing (structural validation should catch this)
        if not src_node or not dst_node:
            continue

        src_op_spec = get_spec(src_node.node_type)
        dst_op_spec = get_spec(dst_node.node_type)

        # If either operator is untyped, we skip port validation for this edge
        if not src_op_spec or not dst_op_spec:
            continue

        # Determine the source output PortSpec
        if "default" in src_op_spec.outputs:
            src_port = src_op_spec.outputs["default"]
        elif len(src_op_spec.outputs) == 1:
            src_port = next(iter(src_op_spec.outputs.values()))
        else:
            # Ambiguous source output - skip validation
            continue

        # 2. Determine the destination input PortSpec
        if edge.dst_port:
            port_name = edge.dst_port
            dst_port = dst_op_spec.inputs.get(port_name)

            if not dst_port:
                # Search for compatible alternatives for a helpful suggestion
                compatible_alternatives = [
                    p
                    for p, ps in dst_op_spec.inputs.items()
                    if is_compatible(src_port.spec, ps.spec)
                ]
                suggestion = ""
                if compatible_alternatives:
                    suggestion = f" Did you mean dst_port='{compatible_alternatives[0]}'?"

                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="E203",
                        node_id=str(edge.dst),
                        message=(
                            f"Node '{edge.dst}' of type '{dst_node.node_type}' "
                            f"has no input port named '{port_name}'.{suggestion}"
                        ),
                    )
                )
                continue
        else:
            # Auto-wiring: If no port is named, find compatible inputs
            compatible_ports = [
                p
                for p, ps in dst_op_spec.inputs.items()
                if is_compatible(src_port.spec, ps.spec)
            ]

            if len(compatible_ports) == 1:
                port_name = compatible_ports[0]
                dst_port = dst_op_spec.inputs[port_name]
            elif len(compatible_ports) > 1:
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="E206",
                        node_id=str(edge.dst),
                        message=(
                            f"Ambiguous connection to '{edge.dst}' ({dst_node.node_type}). "
                            f"Multiple compatible ports found: {compatible_ports}. "
                            "Please specify 'dst_port' explicitly."
                        ),
                    )
                )
                continue
            else:
                # No compatible ports found - structural/port check will handle this if required
                continue

        # Check compatibility between Specs
        src_type = src_port.spec
        dst_type = dst_port.spec

        if not is_compatible(src_type, dst_type):
            conn_path = f"{dst_node.node_id}.{port_name} <- {src_node.node_id}"

            # Detailed Schema Validation for clearer field-level errors
            if isinstance(src_type, Schema) and isinstance(dst_type, Schema):
                src_map = src_type.get_field_map()
                dst_map = dst_type.get_field_map()

                # Check for missing fields (E310)
                for field_name in dst_map:
                    if field_name not in src_map:
                        report.add(
                            ValidationIssue(
                                severity=SEVERITY_ERROR,
                                code="E310",
                                node_id=str(edge.dst),
                                message=(
                                    f"{conn_path}\n"
                                    f"Field '{field_name}' missing from schema.\n"
                                    f"Expected in {port_name}: {dst_map[field_name].dtype}{list(dst_map[field_name].shape)}"
                                ),
                            )
                        )

                # Check for field mismatches (E311)
                for field_name in set(src_map) & set(dst_map):
                    s_spec = src_map[field_name]
                    d_spec = dst_map[field_name]
                    if s_spec.dtype != d_spec.dtype or s_spec.shape != d_spec.shape:
                        report.add(
                            ValidationIssue(
                                severity=SEVERITY_ERROR,
                                code="E311",
                                node_id=str(edge.dst),
                                message=(
                                    f"{conn_path}\n"
                                    f"Field '{field_name}' mismatch:\n"
                                    f"Expected: {d_spec.dtype}{list(d_spec.shape)}\n"
                                    f"Got:      {s_spec.dtype}{list(s_spec.shape)}"
                                ),
                            )
                        )
            else:
                # Search for compatible alternatives on other ports
                compatible_alternatives = [
                    p
                    for p, ps in dst_op_spec.inputs.items()
                    if is_compatible(src_type, ps.spec) and p != port_name
                ]
                suggestion = ""
                if compatible_alternatives:
                    suggestion = f"\n\nSuggestion: Use dst_port='{compatible_alternatives[0]}'"

                # Generic Port Mismatch (E204)
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="E204",
                        node_id=str(edge.dst),
                        message=(
                            f"{conn_path}\n\n"
                            f"Expected:\n{format_spec(dst_type)}\n\n"
                            f"Got:\n{format_spec(src_type)}{suggestion}"
                        ),
                    )
                )

    # 2. Check for missing required inputs
    for nid, node in graph.nodes.items():
        spec = get_spec(node.node_type)
        if not spec:
            continue

        # Collect all connected ports for this node
        connected_ports = set()
        for edge in graph.edges:
            if edge.dst == nid:
                if edge.dst_port:
                    connected_ports.add(edge.dst_port)
                else:
                    # Use auto-wiring logic to see which port is connected
                    src_node = graph.nodes.get(edge.src)
                    if src_node:
                        src_op_spec = get_spec(src_node.node_type)
                        if src_op_spec:
                            # Determine source output spec
                            if "default" in src_op_spec.outputs:
                                src_port = src_op_spec.outputs["default"]
                            elif len(src_op_spec.outputs) == 1:
                                src_port = next(iter(src_op_spec.outputs.values()))
                            else:
                                src_port = None

                            if src_port:
                                # Find compatible ports on this node
                                compatible = [
                                    p
                                    for p, ps in spec.inputs.items()
                                    if is_compatible(src_port.spec, ps.spec)
                                ]
                                if len(compatible) == 1:
                                    connected_ports.add(compatible[0])

        for port_name, port_spec in spec.inputs.items():
            if port_spec.required and port_name not in connected_ports:
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="E205",
                        node_id=nid,
                        message=f"Node '{nid}' missing required input port '{port_name}'",
                    )
                )

    return report
