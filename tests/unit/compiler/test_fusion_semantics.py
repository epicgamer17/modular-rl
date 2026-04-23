import pytest
import torch
from core.graph import Graph
from compiler.compiler import compile_graph
from runtime.executor import execute, register_operator
from runtime.specs import register_spec, OperatorSpec
from core.schema import TensorSpec

pytestmark = pytest.mark.unit

def test_fusion_semantics_numerical_comparison() -> None:
    """Verifies that fusion preserves numerical outputs."""
    # Define OpA (+1), OpB (*2), and FusedOp ( (x+1)*2 )
    def op_a_func(node, inputs, context):
        val = next(iter(inputs.values()))
        return val + 1
        
    def op_b_func(node, inputs, context):
        val = next(iter(inputs.values()))
        return val * 2
        
    def op_fused_func(node, inputs, context):
        val = next(iter(inputs.values()))
        return (val + 1) * 2

    register_operator("SemanticOpA", op_a_func)
    register_operator("SemanticOpB", op_b_func)
    register_operator("SemanticOpFused", op_fused_func)
    register_operator("SemanticSource", lambda node, inputs, context: inputs.get("val", torch.tensor([0.0])))
    register_operator("SemanticSink", lambda node, inputs, context: next(iter(inputs.values())))
    
    spec = TensorSpec(shape=(1,), dtype="float32")
    register_spec("SemanticOpA", OperatorSpec.create(name="SemanticOpA", inputs={"in": spec}, outputs=spec, pure=True, deterministic=True))
    register_spec("SemanticOpB", OperatorSpec.create(name="SemanticOpB", inputs={"in": spec}, outputs=spec, pure=True, deterministic=True))
    register_spec("SemanticOpFused", OperatorSpec.create(name="SemanticOpFused", inputs={"in": spec}, outputs=spec, pure=True, deterministic=True))
    register_spec("SemanticSource", OperatorSpec.create(name="SemanticSource", outputs=spec, pure=True, deterministic=True))
    register_spec("SemanticSink", OperatorSpec.create(name="SemanticSink", inputs={"in": spec}))
    
    from compiler.optimizer import OPTIMIZER_ENGINE
    from compiler.rewrite import FusionRule
    OPTIMIZER_ENGINE.add_rule(FusionRule(name="semantic_fuse", pattern=["SemanticOpA", "SemanticOpB"], replacement="SemanticOpFused"))
    
    # Define Graph
    g = Graph()
    g.add_node("src", "SemanticSource")
    g.add_node("a", "SemanticOpA")
    g.add_node("b", "SemanticOpB")
    g.add_node("sink", "SemanticSink")
    g.add_edge("src", "a", dst_port="in")
    g.add_edge("a", "b", dst_port="in")
    g.add_edge("b", "sink", dst_port="in")
    
    input_data = {"src": torch.tensor([5.0])}
    
    # 1. Run Original Graph (Without optimization)
    compiled_orig = compile_graph(g, optimize=False)
    res_orig = execute(compiled_orig, input_data)
    val_orig = res_orig["sink"]
    
    # 2. Run Optimized Graph
    compiled_opt = compile_graph(g, optimize=True)
    res_opt = execute(compiled_opt, input_data)
    val_opt = res_opt["sink"]
    
    # 3. Compare
    from runtime.values import Value
    v_orig = val_orig.data if isinstance(val_orig, Value) else val_orig
    v_opt = val_opt.data if isinstance(val_opt, Value) else val_opt
    
    assert torch.allclose(v_orig, v_opt), f"Numerical mismatch: {v_orig} != {v_opt}"
    assert v_orig.item() == 12.0
