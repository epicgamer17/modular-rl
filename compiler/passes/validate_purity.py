from core.graph import Graph
from compiler.validation import (
    ValidationReport,
    ValidationIssue,
    SEVERITY_ERROR,
    SEVERITY_WARN,
)
from runtime.specs import get_spec


def validate_purity(graph: Graph, context: str = "both") -> ValidationReport:
    """
    Validates node purity and side effects.
    
    Checks:
    - Side-effect nodes are not duplicated accidentally (D001).
    - Nodes are allowed in the current context (actor/learner) (D002).
    - Resource constraints (e.g. optimizer in actor) (D003).
    """
    report = ValidationReport()
    side_effects_found = {}  # effect_name -> node_id

    for nid, node in graph.nodes.items():
        spec = get_spec(node.node_type)
        if not spec:
            continue

        # 1. Check for duplicated side effects
        for effect in spec.side_effects:
            if effect in side_effects_found:
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_WARN,
                        code="D001",
                        node_id=nid,
                        message=(
                            f"Side effect '{effect}' is duplicated. "
                            f"Already present in node '{side_effects_found[effect]}'. "
                            "This may cause unintended behavior or multiple updates."
                        ),
                    )
                )
            else:
                side_effects_found[effect] = nid

        # 2. Check context violations
        if context != "both":
            allowed = spec.allowed_contexts
            if context not in allowed and "both" not in allowed:
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="D002",
                        node_id=nid,
                        message=(
                            f"Node '{nid}' of type '{node.node_type}' is not allowed "
                            f"in '{context}' context. (Allowed: {list(allowed)})"
                        ),
                    )
                )

        # 3. Resource violations (Optimizer in Actor)
        if context == "actor" and spec.requires_optimizer:
            report.add(
                ValidationIssue(
                    severity=SEVERITY_ERROR,
                    code="D003",
                    node_id=nid,
                    message=(
                        f"Node '{nid}' of type '{node.node_type}' requires an optimizer, "
                        "which is not allowed in 'actor' context."
                    ),
                )
            )

    return report
