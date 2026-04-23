import pytest
from core.graph import (
    Graph,
    NODE_TYPE_REPLAY_QUERY,
    NODE_TYPE_TARGET_SYNC,
    NODE_TYPE_EXPLORATION,
    NODE_TYPE_METRICS_SINK,
)
from core.schema import TAG_ON_POLICY
from compiler.passes.validate_rl import validate_rl_semantics
from compiler.validation import SEVERITY_ERROR

pytestmark = pytest.mark.unit


def test_rl_on_policy_with_replay_fails() -> None:
    """Rule R001: On-policy node cannot consume replay buffer."""
    g = Graph()
    g.add_node("replay", NODE_TYPE_REPLAY_QUERY)
    # Mark PPO loss as on-policy
    g.add_node("ppo_loss", "PPOLoss", tags=[TAG_ON_POLICY])
    g.add_edge("replay", "ppo_loss")

    report = validate_rl_semantics(g)
    assert report.has_errors()
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(i.code == "R001" and "ppo_loss" in i.message for i in issues)


def test_rl_exploration_missing_q_fails() -> None:
    """Rule R002: Exploration requires q_values input."""
    g = Graph()
    g.add_node("explore", NODE_TYPE_EXPLORATION)
    # No incoming q_values port - should fail

    report = validate_rl_semantics(g)
    assert report.has_errors()
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(i.code == "R002" and "explore" in i.message for i in issues)


def test_rl_metrics_sink_as_input_fails() -> None:
    """Rule R003: MetricsSink cannot have successors."""
    g = Graph()
    g.add_node("metrics", NODE_TYPE_METRICS_SINK)
    g.add_node("downstream", "SomeNode")
    g.add_edge("metrics", "downstream")

    report = validate_rl_semantics(g)
    assert report.has_errors()
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(i.code == "R003" and "metrics" in i.message for i in issues)


def test_rl_target_sync_disconnected_fails() -> None:
    """Rule R004: TargetSync requires inputs from online networks."""
    g = Graph()
    g.add_node("sync", NODE_TYPE_TARGET_SYNC)
    # Disconnected TargetSync node

    report = validate_rl_semantics(g)
    assert report.has_errors()
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(i.code == "R004" and "sync" in i.message for i in issues)


def test_rl_optimizer_updating_target_fails() -> None:
    """Rule R005: Optimizer cannot update target parameters directly."""
    g = Graph()
    # Node with 'Optimizer' in type and a 'target' parameter
    g.add_node("opt", "AdamOptimizer", params={"target_net_weights": "params_ref"})

    report = validate_rl_semantics(g)
    assert report.has_errors()
    issues = report.get_issues_by_severity(SEVERITY_ERROR)
    assert any(i.code == "R005" and "target_net_weights" in i.message for i in issues)


def test_rl_valid_policy_passes() -> None:
    """Verifies that a correctly wired RL path passes validation."""
    g = Graph()
    g.add_node("q_values", "QNetwork")
    g.add_node("explore", NODE_TYPE_EXPLORATION)
    g.add_edge("q_values", "explore", dst_port="q_values")

    report = validate_rl_semantics(g)
    assert not report.has_errors()
