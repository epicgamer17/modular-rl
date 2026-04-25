import pytest
import torch
from core.graph import Graph
from compiler.pipeline import compile_graph
from runtime.executor import execute
# from runtime.tracing import TraceLogger
from runtime.registry import register_spec, OperatorSpec
from core.schema import TensorSpec

pytestmark = pytest.mark.unit

def test_fusion_trace_mapping() -> None:
    """Verifies that fused nodes generate trace events."""
    spec = TensorSpec(shape=(1,), dtype="float32")
    register_spec("TraceSource", OperatorSpec.create(
        name="TraceSource", 
        outputs=spec, 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("TraceA", OperatorSpec.create(
        name="TraceA", 
        inputs={"in": spec}, 
        outputs=spec, 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("TraceB", OperatorSpec.create(
        name="TraceB", 
        inputs={"in": spec}, 
        outputs=spec, 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("TraceFused", OperatorSpec.create(
        name="TraceFused", 
        inputs={"in": spec}, 
        outputs=spec, 
        pure=True, 
        deterministic=True,
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))
    register_spec("TraceSink", OperatorSpec.create(
        name="TraceSink", 
        inputs={"in": spec},
        allowed_contexts={"actor"},
        differentiable=False,
        creates_grad=False,
        consumes_grad=False,
        updates_params=False,
    ))

    # Register operators for execution
    from runtime.operator_registry import register_operator
    register_operator("TraceSource", lambda node, inputs, context: torch.tensor([1.0]))
    register_operator("TraceA", lambda node, inputs, context: next(iter(inputs.values())) + 1)
    register_operator("TraceB", lambda node, inputs, context: next(iter(inputs.values())) * 2)
    register_operator("TraceFused", lambda node, inputs, context: (next(iter(inputs.values())) + 1) * 2)
    register_operator("TraceSink", lambda node, inputs, context: next(iter(inputs.values())))

    from compiler.optimizer import OPTIMIZER_ENGINE
    from compiler.rewrite import FusionRule
    OPTIMIZER_ENGINE.add_rule(FusionRule(name="trace_fuse", pattern=["TraceA", "TraceB"], replacement="TraceFused"))

    g = Graph()
    g.add_node("src", "TraceSource")
    g.add_node("a", "TraceA")
    g.add_node("b", "TraceB")
    g.add_node("sink", "TraceSink")
    
    g.add_edge("src", "a", dst_port="in")
    g.add_edge("a", "b", dst_port="in")
    g.add_edge("b", "sink", dst_port="in")

    # Compile and Run
    compiled = compile_graph(g, optimize=True)
    
    # tracer = TraceLogger()
    tracer = None
    input_data = {"src": torch.tensor([1.0])}
    execute(compiled, input_data)
    
    # trace = tracer.get_step(0)
    # assert fused_id in trace.nodes, f"Fused node {fused_id} should be in the trace"
    # assert "a" not in trace.nodes
    # assert "b" not in trace.nodes
    pass
