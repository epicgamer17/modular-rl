"""
RL Semantic validation pass.
Enforces RL-specific architectural constraints on the computation graph.
"""

from core.graph import (
    Graph,
    NODE_TYPE_REPLAY_QUERY,
    NODE_TYPE_TARGET_SYNC,
    NODE_TYPE_EXPLORATION,
    NODE_TYPE_METRICS_SINK,
)
from core.schema import TAG_ON_POLICY
from compiler.validation import (
    ValidationIssue,
    ValidationReport,
    SEVERITY_ERROR,
)


def validate_rl_semantics(graph: Graph) -> ValidationReport:
    """
    Performs RL-specific semantic validation.

    Checks:
    1. On-policy nodes (TAG_ON_POLICY) cannot consume data from ReplayQuery (R001).
    2. Exploration nodes must have a 'q_values' input (R002).
    3. MetricsSink nodes should not feed into other computations (R003).
    4. TargetSync nodes should be connected to a learner path (R004).
    5. Target networks should not be updated by optimizers directly (R005).

    Args:
        graph: The Graph instance to validate.

    Returns:
        A ValidationReport containing any discovered issues.
    """
    report = ValidationReport()

    # 1. Rule: On-policy node cannot consume replay query
    # On-policy algorithms (like PPO) should not use off-policy replay data.
    replay_queries = [
        nid
        for nid, node in graph.nodes.items()
        if node.node_type == NODE_TYPE_REPLAY_QUERY
    ]
    for start_node in replay_queries:
        visited = set()
        stack = [start_node]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)

            node = graph.nodes[u]
            if TAG_ON_POLICY in node.tags:
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="R001",
                        node_id=str(u),
                        message=f"On-policy node '{u}' cannot consume data from ReplayQuery '{start_node}'",
                    )
                )

            # Traverse downstream
            for v in graph.adjacency_list.get(u, set()):
                if v in graph.nodes:
                    stack.append(v)

    # 2. Rule: Exploration requires q_values input
    # Standard exploration operators (like Epsilon-Greedy) expect Q-values.
    for nid, node in graph.nodes.items():
        if node.node_type == NODE_TYPE_EXPLORATION:
            incoming_ports = {edge.dst_port for edge in graph.edges if edge.dst == nid}
            if "q_values" not in incoming_ports:
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="R002",
                        node_id=str(nid),
                        message=f"Exploration node '{nid}' is missing the required 'q_values' input port",
                    )
                )

    # 3. Rule: MetricsSink cannot feed training path
    # Metrics sinks are terminal and should not be used as inputs for other logic.
    for nid, node in graph.nodes.items():
        if node.node_type == NODE_TYPE_METRICS_SINK:
            if graph.adjacency_list.get(nid):
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="R003",
                        node_id=str(nid),
                        message=f"MetricsSink '{nid}' cannot have successor nodes; it should be a terminal sink",
                    )
                )

    # 4. Rule: Target sync requires learner path
    # TargetSync needs an online network as input to sync to the target.
    for nid, node in graph.nodes.items():
        if node.node_type == NODE_TYPE_TARGET_SYNC:
            has_inputs = any(edge.dst == nid for edge in graph.edges)
            if not has_inputs:
                report.add(
                    ValidationIssue(
                        severity=SEVERITY_ERROR,
                        code="R004",
                        node_id=str(nid),
                        message=f"TargetSync '{nid}' is disconnected; it requires an online network or learner path input",
                    )
                )

    # 5. Rule: Target net cannot be optimizer-updated directly
    # Target networks should only be updated via TargetSync from an online network.
    for nid, node in graph.nodes.items():
        # Detect optimizer-like nodes by type or tags
        is_optimizer = "Optimizer" in node.node_type or "optimizer" in node.tags
        if is_optimizer:
            for param_key in node.params:
                if "target" in param_key.lower():
                    report.add(
                        ValidationIssue(
                            severity=SEVERITY_ERROR,
                            code="R005",
                            node_id=str(nid),
                            message=(
                                f"Optimizer '{nid}' is attempting to directly update "
                                f"a target network parameter '{param_key}'"
                            ),
                        )
                    )

    return report
