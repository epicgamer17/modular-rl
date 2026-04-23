import pytest
import torch
from core.graph import Graph, NODE_TYPE_SOURCE, NODE_TYPE_SINK
from compiler.optimizer import optimize_graph, dead_node_elimination, OptimizationReport, OPTIMIZER_ENGINE
from compiler.rewrite import FusionRule
from runtime.specs import register_spec, OperatorSpec, SingleObs, SingleQ, Scalar

pytestmark = pytest.mark.unit

def test_optimizer_report_fusion():
    """Verifies that the optimization report correctly records node fusion."""
    # Register dummy specs for testing
    register_spec("QValuesSingle", OperatorSpec.create("QValuesSingle", inputs={"obs": SingleObs}, outputs={"q_values": SingleQ}, pure=True, deterministic=True))
    register_spec("Argmax", OperatorSpec.create("Argmax", inputs={"input": SingleQ}, outputs={"output": Scalar("int64")}, pure=True, deterministic=True))
    register_spec("GreedyPolicy", OperatorSpec.create("GreedyPolicy", inputs={"obs": SingleObs}, outputs={"action": Scalar("int64")}, pure=True, deterministic=True))
    register_spec("ReportSink", OperatorSpec.create("ReportSink", inputs={"default": Scalar("int64")}, outputs={}, pure=False))

    graph = Graph()
    graph.add_node("obs", NODE_TYPE_SOURCE)
    graph.add_node("q", "QValuesSingle")
    graph.add_node("arg", "Argmax")
    graph.add_node("out", "ReportSink")

    graph.add_edge("obs", "q", dst_port="obs")
    graph.add_edge("q", "arg", dst_port="input")
    graph.add_edge("arg", "out", dst_port="default")

    report = OptimizationReport()
    optimized_graph = optimize_graph(graph, report=report)

    # Verify report contents
    assert len(report.steps) > 0
    step = report.steps[0]
    assert step.rule_name == "greedy_policy"
    assert "QValuesSingle" in step.pattern
    assert "Argmax" in step.pattern
    assert step.replacement == "GreedyPolicy"
    assert "q" in step.removed_nodes
    assert "arg" in step.removed_nodes
    
    # Verify string representation
    report_str = str(report)
    assert "Applied rule greedy_policy:" in report_str
    assert "[QValuesSingle -> Argmax] => [GreedyPolicy]" in report_str
    assert "Removed nodes: q, arg" in report_str

def test_optimizer_report_dead_node():
    """Verifies that the optimization report correctly records dead node elimination."""
    graph = Graph()
    graph.add_node("obs", NODE_TYPE_SOURCE)
    graph.add_node("dead", "QValuesSingle") # No path to sink

    report = OptimizationReport()
    optimized_graph = dead_node_elimination(graph, report=report)

    assert "dead" in report.dead_nodes_removed
    
    report_str = str(report)
    # Both obs and dead should be removed as obs only leads to dead
    assert "Dead Node Elimination: Removed 2 nodes" in report_str
    assert "Nodes: obs, dead" in report_str or "Nodes: dead, obs" in report_str
