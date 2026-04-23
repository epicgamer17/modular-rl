"""
Main Compiler entry point for the RL IR.
Orchestrates validation passes and optimization.
"""

from typing import Set, Optional
from core.graph import Graph
from compiler.validation import ValidationReport, SEVERITY_ERROR, SEVERITY_WARN
from compiler.passes.validate_structure import validate_structure
from compiler.passes.validate_ports import validate_ports
from compiler.passes.validate_rl import validate_rl_semantics
from compiler.passes.validate_handles import validate_handles


def compile_graph(
    graph: Graph,
    strict: bool = False,
    model_handles: Optional[Set[str]] = None,
    buffer_handles: Optional[Set[str]] = None,
) -> Graph:
    """
    Compiles an RL IR graph by running all validation passes.

    Args:
        graph: The graph to compile.
        strict: If True, treats all warnings as errors.
        model_handles: Set of registered model handles in the target environment.
        buffer_handles: Set of registered buffer handles in the target environment.

    Returns:
        The validated graph.

    Raises:
        RuntimeError: If validation fails (due to errors or strict mode warnings).
    """
    report = ValidationReport()

    # 1. Structural Checks (Required, Cycles, unreachable nodes)
    report.merge(validate_structure(graph))

    # 2. Port Contract Checks (Compatible types, Schema field audit)
    report.merge(validate_ports(graph))

    # 3. RL Semantic Checks (On-policy vs Off-policy, sync rules)
    report.merge(validate_rl_semantics(graph))

    # 4. Handle Registry Checks (Model/Buffer handle existence)
    report.merge(validate_handles(graph, model_handles, buffer_handles))

    # Check for hard errors
    if report.has_errors():
        issues = report.get_issues_by_severity(SEVERITY_ERROR)
        error_details = "\n".join(
            [f"[{i.code}] Node {i.node_id}: {i.message}" for i in issues]
        )
        raise RuntimeError(f"Graph compilation failed with errors:\n{error_details}")

    # Strict mode: treat all warnings as fatal errors
    if strict and report.has_warnings():
        issues = report.get_issues_by_severity(SEVERITY_WARN)
        warn_details = "\n".join(
            [f"[{i.code}] Node {i.node_id}: {i.message}" for i in issues]
        )
        raise RuntimeError(
            f"Graph compilation failed in STRICT mode due to warnings:\n{warn_details}"
        )

    return graph
