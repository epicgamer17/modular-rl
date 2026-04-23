import torch
import torch.nn as nn
from core.graph import Graph
from compiler.validation import (
    ValidationReport,
    ValidationIssue,
    SEVERITY_ERROR,
)
from runtime.state import ReplayBuffer

def validate_ir_purity(graph: Graph) -> ValidationReport:
    """
    Validates that the IR remains 'pure' and serializable by ensuring node parameters
    do not contain live runtime objects or non-serializable state.

    Rules:
    - P001: No nn.Module in params
    - P002: No optimizer instance in params
    - P003: No ReplayBuffer in params
    - P004: No callable closures/lambdas in params
    """
    report = ValidationReport()

    for nid, node in graph.nodes.items():
        for param_name, param_value in node.params.items():
            # Rule P001: No nn.Module
            if isinstance(param_value, nn.Module):
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="P001",
                        node_id=nid,
                        message=(
                            f"Node parameter '{param_name}' contains a live torch.nn.Module "
                            f"({type(param_value).__name__}). Use a model handle string instead."
                        ),
                    )
                )

            # Rule P002: No optimizer instance
            elif isinstance(param_value, torch.optim.Optimizer):
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="P002",
                        node_id=nid,
                        message=(
                            f"Node parameter '{param_name}' contains a live torch.optim.Optimizer "
                            f"({type(param_value).__name__}). Use an optimizer handle string instead."
                        ),
                    )
                )

            # Rule P003: No ReplayBuffer
            elif isinstance(param_value, ReplayBuffer):
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="P003",
                        node_id=nid,
                        message=(
                            f"Node parameter '{param_name}' contains a live ReplayBuffer instance. "
                            "Use a buffer ID string instead."
                        ),
                    )
                )

            # Rule P004: No callable closures/lambdas
            elif callable(param_value) and not isinstance(param_value, type):
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="P004",
                        node_id=nid,
                        message=(
                            f"Node parameter '{param_name}' contains a callable closure or lambda. "
                            "Parameters must be serializable primitives or handles."
                        ),
                    )
                )

    return report
