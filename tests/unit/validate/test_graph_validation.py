import io
import sys
from core.graph import Graph
from core.schema import TensorSpec, Schema, Field
from agents.ppo.operators import register_ppo_operators
from validate.graph_validator import validate_graph

def setup_operators():
    register_ppo_operators()

class AssertRaises:
    def __init__(self, exception_type, match=None):
        self.exception_type = exception_type
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"{self.exception_type.__name__} not raised")
        if not issubclass(exc_type, self.exception_type):
            raise AssertionError(f"Expected {self.exception_type.__name__}, got {exc_type.__name__}")
        if self.match and self.match not in str(exc_val):
             raise AssertionError(f"Error message '{str(exc_val)}' does not match '{self.match}'")
        return True

def test_missing_input_port():
    """Verify that missing required ports trigger a compilation error."""
    graph = Graph()
    graph.add_node("src", "Source")
    graph.add_node("val_loss", "ValueLoss")
    graph.add_edge("src", "val_loss", src_port="values", dst_port="values")
    
    # Missing 'returns' port
    with AssertRaises(ValueError, match="is missing required input ports"):
        validate_graph(graph)

def test_shape_mismatch():
    """Verify that shape mismatches between nodes trigger an error."""
    graph = Graph()
    graph.add_node("src", "Source", schema_out=Schema(fields=[
        Field("logits", TensorSpec(shape=(32, 2), dtype="float32")),
        Field("returns", TensorSpec(shape=(32,), dtype="float32"))
    ]))
    graph.add_node("val_loss", "ValueLoss")
    
    # WRONG: feeding logits [32, 2] into values [B]
    graph.add_edge("src", "val_loss", src_port="logits", dst_port="values")
    graph.add_edge("src", "val_loss", src_port="returns", dst_port="returns")
    
    with AssertRaises(ValueError, match="Shape Mismatch"):
        validate_graph(graph)

def test_side_effect_violation():
    """Verify that stateful operators are rejected in Actor graphs."""
    graph = Graph()
    graph.tags = ["Actor"]
    graph.add_node("loss_src", "Source")
    graph.add_node("opt", "PPO_Optimizer")
    graph.add_edge("loss_src", "opt", dst_port="loss")
    
    with AssertRaises(ValueError, match="Side-effect violation in Actor graph"):
        validate_graph(graph)

def test_domain_mismatch_warning():
    """Verify that using operators from a different algorithm family warns."""
    graph = Graph()
    graph.tags = ["DQN"]
    graph.add_node("x_src", "Source")
    graph.add_node("clip_node", "Clip")
    graph.add_edge("x_src", "clip_node", dst_port="x")
    
    captured_output = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured_output
    try:
        validate_graph(graph)
        output = captured_output.getvalue()
        assert "LINT WARNING" in output
        assert "Clip" in output
    finally:
        sys.stdout = original_stdout

def test_valid_ppo_graph():
    """Verify that a correctly wired PPO graph passes all checks."""
    graph = Graph()
    graph.tags = ["PPO"]
    graph.add_node("src", "Source", schema_out=Schema(fields=[
        Field("values", TensorSpec(shape=(32,), dtype="float32")),
        Field("returns", TensorSpec(shape=(32,), dtype="float32"))
    ]))
    graph.add_node("val_loss", "ValueLoss")
    graph.add_edge("src", "val_loss", src_port="values", dst_port="values")
    graph.add_edge("src", "val_loss", src_port="returns", dst_port="returns")
    
    # Should pass all passes
    validate_graph(graph)

def test_math_category_violation():
    """Verify that training-only categories (loss) are rejected in Actor graphs."""
    graph = Graph()
    graph.tags = ["Actor"]
    graph.add_node("src", "Source")
    graph.add_node("loss", "ValueLoss")
    graph.add_edge("src", "loss", src_port="values", dst_port="values")
    
    with AssertRaises(ValueError, match="is a loss node, which is illegal in inference loops"):
        validate_graph(graph)

if __name__ == "__main__":
    setup_operators()
    print("Running test_missing_input_port...")
    test_missing_input_port()
    print("Running test_shape_mismatch...")
    test_shape_mismatch()
    print("Running test_side_effect_violation...")
    test_side_effect_violation()
    print("Running test_math_category_violation...")
    test_math_category_violation()
    print("Running test_domain_mismatch_warning...")
    test_domain_mismatch_warning()
    print("Running test_valid_ppo_graph...")
    test_valid_ppo_graph()
    print("All tests passed!")
