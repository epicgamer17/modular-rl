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

    Args:
        graph: The Graph instance to validate.

    Returns:
        A ValidationReport containing any discovered issues.
    """
    report = ValidationReport()

    for edge in graph.edges:
        src_node = graph.nodes.get(edge.src)
        dst_node = graph.nodes.get(edge.dst)

        # Skip if nodes are missing (structural validation should catch this)
        if not src_node or not dst_node:
            continue

        src_spec = get_spec(src_node.node_type)
        dst_spec = get_spec(dst_node.node_type)

        # If either operator is untyped, we skip port validation for this edge
        if not src_spec or not dst_spec:
            continue

        # 1. Determine the source output spec.
        # Since Edge currently only tracks dst_port, we assume the primary output
        # is either 'default' or the only available output.
        if "default" in src_spec.outputs:
            src_type = src_spec.outputs["default"]
        elif len(src_spec.outputs) == 1:
            src_type = next(iter(src_spec.outputs.values()))
        else:
            # Ambiguous source output - skip validation
            continue

        # 2. Determine the destination input port and its spec.
        if edge.dst_port:
            port_name = edge.dst_port
        else:
            # If no port is named, and there's only one input, assume that's the target.
            if len(dst_spec.inputs) == 1:
                port_name = next(iter(dst_spec.inputs.keys()))
            else:
                # Ambiguous destination port - skip validation
                continue

        dst_type = dst_spec.inputs.get(port_name)
        if not dst_type:
            report.add(
                ValidationIssue(
                    severity=SEVERITY_ERROR,
                    code="E203",
                    node_id=str(edge.dst),
                    message=(
                        f"Node '{edge.dst}' of type '{dst_node.node_type}' "
                        f"has no input port named '{port_name}'"
                    ),
                )
            )
            continue

        # 3. Check compatibility between source output and destination input.
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
                # Generic Port Mismatch (E204)
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="E204",
                        node_id=str(edge.dst),
                        message=(
                            f"{conn_path}\n\n"
                            f"Expected:\n{format_spec(dst_type)}\n\n"
                            f"Got:\n{format_spec(src_type)}"
                        ),
                    )
                )

    return report
