from core.graph import Graph
from compiler.validation import ValidationReport, ValidationIssue, SEVERITY_ERROR
from runtime.registry import get_spec

def validate_domains(graph: Graph) -> ValidationReport:
    """
    Validates domain-specific constraints (e.g., policy gradient vs Q-learning).
    
    Checks:
    - Mixing incompatible RL domains in the same graph (E501).
    """
    report = ValidationReport()
    
    # Domains can be identified via spec.domain_tags
    all_domains = set()
    for node in graph.nodes.values():
        spec = get_spec(node.node_type)
        if spec and spec.domain_tags:
            all_domains.update(spec.domain_tags)
            
    # Example rule: can't mix 'policy_gradient' and 'q_learning' (simplified)
    if "policy_gradient" in all_domains and "q_learning" in all_domains:
        report.add(ValidationIssue(
            severity=SEVERITY_ERROR,
            code="E501",
            node_id=None,
            message="Graph contains both Policy Gradient and Q-Learning nodes. Domain mixing is restricted."
        ))

    # 2. On-Policy vs Replay Buffer semantic check
    has_on_policy = any("OnPolicy" in node.tags for node in graph.nodes.values())
    if has_on_policy:
        for nid, node in graph.nodes.items():
            if "Replay" in node.tags:
                report.add(ValidationIssue(
                    severity=SEVERITY_ERROR,
                    code="R001",
                    node_id=nid,
                    message=f"Semantic error: OnPolicy graph cannot consume data from Replay buffer node '{nid}'."
                ))

    return report
