"""
Validation pass for string handles (models, buffers).
Ensures that all referenced handles are registered in the runtime environment.
"""

from typing import Set, Optional
from core.graph import (
    Graph,
    NODE_TYPE_REPLAY_QUERY,
    NODE_TYPE_TARGET_SYNC,
)
from compiler.validation import (
    ValidationIssue,
    ValidationReport,
    SEVERITY_ERROR,
)


def validate_handles(
    graph: Graph,
    model_handles: Optional[Set[str]] = None,
    buffer_handles: Optional[Set[str]] = None,
) -> ValidationReport:
    """
    Validates that all model and buffer handles referenced in the graph exist.

    Args:
        graph: The Graph instance to validate.
        model_handles: Set of available model handle strings.
        buffer_handles: Set of available buffer handle strings.

    Returns:
        A ValidationReport containing any discovered issues.
    """
    report = ValidationReport()
    model_handles = model_handles or set()
    buffer_handles = buffer_handles or set()

    for nid, node in graph.nodes.items():
        # 1. Check Model Handles
        # TargetSync nodes explicitly reference source and target models by handle.
        if node.node_type == NODE_TYPE_TARGET_SYNC:
            for key in ["source_handle", "target_handle"]:
                if key in node.params:
                    handle = node.params[key]
                    if handle not in model_handles:
                        report.add(
                            ValidationIssue(
                                severity=SEVERITY_ERROR,
                                code="H001",
                                node_id=str(nid),
                                message=f"Unknown model handle: '{handle}'",
                            )
                        )

        # Other nodes (like QNetwork or Actor) may use 'model_handle' parameter.
        if "model_handle" in node.params:
            handle = node.params["model_handle"]
            if handle not in model_handles:
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="H001",
                        node_id=str(nid),
                        message=f"Unknown model handle: '{handle}'",
                    )
                )

        # 2. Check Buffer Handles
        # ReplayQuery nodes reference buffers via 'buffer_id' (defaults to 'main').
        if node.node_type == NODE_TYPE_REPLAY_QUERY:
            buffer_id = node.params.get("buffer_id", "main")
            if buffer_id not in buffer_handles:
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="H002",
                        node_id=str(nid),
                        message=f"Unknown buffer handle: '{buffer_id}'",
                    )
                )

    return report
