import pytest
import torch
from core.graph import Graph, NODE_TYPE_SOURCE, NODE_TYPE_SINK
from compiler.optimizer import optimize_graph, OptimizationReport, OPTIMIZER_ENGINE
from compiler.passes.autodiff import autodiff
from compiler.passes.collect_trainable_parameters import collect_trainable_parameters
from compiler.rewrite import FusionRule
from runtime.specs import register_spec, OperatorSpec, SingleQ, Scalar

pytestmark = pytest.mark.unit

def test_training_report_contains_backward():
    """
    Verifies that the optimization report includes all required explainability sections:
    - Trainable parameters
    - Backward pass insertion
    - Fusion skips due to backward boundaries
    - no_grad hoisting for target networks
    """
    # 1. Setup specs for a standard training graph
    register_spec("QValues", OperatorSpec.create(
        "QValues", 
        inputs={}, 
        outputs={"q": SingleQ}, 
        parameter_handles=["model_handle"],
        pure=True,
        deterministic=True
    ))
    register_spec("TargetQ", OperatorSpec.create(
        "TargetQ", 
        inputs={}, 
        outputs={"q": SingleQ}, 
        parameter_handles=["target_handle"],
        pure=True,
        deterministic=True
    ))
    register_spec("IdentityBoundary", OperatorSpec.create(
        "IdentityBoundary",
        inputs={"input": SingleQ},
        outputs={"output": SingleQ},
        creates_grad=True,
        pure=True,
        deterministic=True
    ))
    register_spec("MSELoss", OperatorSpec.create(
        "MSELoss", 
        inputs={"input": SingleQ, "target": SingleQ}, 
        outputs={"loss": Scalar("float32")}, 
        creates_grad=True,
        pure=True,
        deterministic=True
    ))
    register_spec("Optimizer", OperatorSpec.create(
        "Optimizer", 
        inputs={"loss": Scalar("float32")}, 
        outputs={}, 
        consumes_grad=True,
        pure=False
    ))
    register_spec("Backward", OperatorSpec.create(
        "Backward", 
        inputs={"loss": Scalar("float32")}, 
        outputs={}, 
        pure=False
    ))
    register_spec("GradBuffer", OperatorSpec.create(
        "GradBuffer", 
        inputs={}, 
        outputs={}, 
        pure=False
    ))

    # 2. Build the graph
    graph = Graph()
    graph.add_node("online_q", "QValues", params={"model_handle": "online_q"})
    graph.add_node("target_q", "TargetQ", params={"target_handle": "target_q"})
    graph.add_node("boundary", "IdentityBoundary")
    graph.add_node("loss", "MSELoss")
    graph.add_node("opt", "Optimizer", params={"model_handle": "online_q"})
    graph.add_node("sink", NODE_TYPE_SINK)

    graph.add_edge("online_q", "boundary", dst_port="input")
    graph.add_edge("boundary", "loss", dst_port="input")
    graph.add_edge("target_q", "loss", dst_port="target")
    graph.add_edge("loss", "opt", dst_port="loss")
    graph.add_edge("opt", "sink")

    # 3. Add a fusion rule that would match across a backward boundary
    boundary_rule = FusionRule(
        name="boundary_fusion_test",
        pattern=["QValues", "IdentityBoundary"],
        replacement="FusedBoundary"
    )
    OPTIMIZER_ENGINE.add_rule(boundary_rule)

    report = OptimizationReport()
    
    # 4. Run compiler passes with report
    collect_trainable_parameters(graph, report=report)
    graph = autodiff(graph, report=report)
    optimized_graph = optimize_graph(graph, report=report)
    
    report_str = str(report)
    
    # 5. Verify the report contents match the requested format
    
    # Trainable params section
    assert "Detected trainable params:" in report_str
    assert "  online_q" in report_str
    
    # Backward pass section
    assert "Inserted backward pass:" in report_str
    assert "  loss -> Backward(online_q)" in report_str
    
    # Skipped fusion section
    assert "Skipped fusion:" in report_str
    assert "  backward boundary blocks fusion" in report_str
    
    # No-grad hoist section
    assert "Applied no_grad hoist:" in report_str
    assert "  target_q branch" in report_str

    # Clean up the test rule to avoid affecting other tests
    OPTIMIZER_ENGINE.rules = [r for r in OPTIMIZER_ENGINE.rules if r.name != "boundary_fusion_test"]
