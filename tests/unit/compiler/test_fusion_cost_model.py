import pytest
from core.graph import Graph, NODE_TYPE_SOURCE, NODE_TYPE_SINK
from compiler.optimizer import optimize_graph, OptimizationReport, OPTIMIZER_ENGINE
from compiler.rewrite import FusionRule, RewriteEngine
from runtime.registry import register_spec, OperatorSpec, SingleObs, SingleQ, Scalar

pytestmark = pytest.mark.unit

def test_fusion_skipped_due_to_profitability():
    """Verifies that fusion is skipped if it doesn't meet the profitability threshold."""
    # Register specs with launch costs
    # OpA + OpB -> OpFused
    # launch_cost: 1 + 1 -> 1.5 (Profit = 0.5)
    register_spec("OpA", OperatorSpec.create("OpA", inputs={"in": Scalar("float32")}, outputs={"out": Scalar("float32")}, pure=True, deterministic=True, kernel_launch_cost=1.0))
    register_spec("OpB", OperatorSpec.create("OpB", inputs={"in": Scalar("float32")}, outputs={"out": Scalar("float32")}, pure=True, deterministic=True, kernel_launch_cost=1.0))
    register_spec("OpFused", OperatorSpec.create("OpFused", inputs={"in": Scalar("float32")}, outputs={"out": Scalar("float32")}, pure=True, deterministic=True, kernel_launch_cost=1.5))
    register_spec("Sink", OperatorSpec.create("Sink", inputs={"in": Scalar("float32")}, outputs={}, pure=False))

    engine = RewriteEngine()
    # Require profit of at least 0.6
    rule = FusionRule(name="test_fusion", pattern=["OpA", "OpB"], replacement="OpFused", min_profitability=0.6)
    engine.add_rule(rule)

    graph = Graph()
    graph.add_node("s", NODE_TYPE_SOURCE)
    graph.add_node("a", "OpA")
    graph.add_node("b", "OpB")
    graph.add_node("out", "Sink")
    graph.add_edge("s", "a", dst_port="in")
    graph.add_edge("a", "b", dst_port="in")
    graph.add_edge("b", "out", dst_port="in")

    report = OptimizationReport()
    # Manually use engine to avoid global engine rules
    optimized_graph = engine.apply(graph, report=report)

    # Fusion should be skipped because profit (0.5) < min_profitability (0.6)
    assert len(report.steps) == 0
    assert len(report.skipped_fusions) == 1
    assert report.skipped_fusions[0]["reason"] == "Not profitable based on cost model"
    assert "a" in report.skipped_fusions[0]["nodes"]
    assert "b" in report.skipped_fusions[0]["nodes"]

def test_fusion_skipped_due_to_branching():
    """Verifies that fusion is skipped if a node in the chain has multiple consumers."""
    # Register specs
    register_spec("OpX", OperatorSpec.create("OpX", inputs={"in": Scalar("float32")}, outputs={"out": Scalar("float32")}, pure=True, deterministic=True, kernel_launch_cost=1.0))
    register_spec("OpY", OperatorSpec.create("OpY", inputs={"in": Scalar("float32")}, outputs={"out": Scalar("float32")}, pure=True, deterministic=True, kernel_launch_cost=1.0))
    register_spec("OpXY", OperatorSpec.create("OpXY", inputs={"in": Scalar("float32")}, outputs={"out": Scalar("float32")}, pure=True, deterministic=True, kernel_launch_cost=1.5))

    engine = RewriteEngine()
    engine.add_rule(FusionRule(name="xy_fusion", pattern=["OpX", "OpY"], replacement="OpXY"))

    graph = Graph()
    graph.add_node("s", NODE_TYPE_SOURCE)
    graph.add_node("x", "OpX")
    graph.add_node("y", "OpY")
    graph.add_node("out1", "Sink")
    graph.add_node("out2", "Sink") # Second consumer of 'x'

    graph.add_edge("s", "x", dst_port="in")
    graph.add_edge("x", "y", dst_port="in") # Path for fusion
    graph.add_edge("x", "out2", dst_port="in") # Second consumer makes x branching
    graph.add_edge("y", "out1", dst_port="in")

    report = OptimizationReport()
    optimized_graph = engine.apply(graph, report=report)

    # Fusion should be skipped by find_linear_chain due to branching at 'x'
    assert len(report.steps) == 0
    # Note: skipped_fusions is only populated if it matches the pattern but fails profitability/constraints
    # Since find_linear_chain skips branching nodes, it won't even be a candidate for fusion
    assert "x" in optimized_graph.nodes
    assert "y" in optimized_graph.nodes
