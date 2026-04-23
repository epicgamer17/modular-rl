import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from core.graph import Graph
from runtime.state import ReplayBuffer
from compiler.passes.validate_ir_purity import validate_ir_purity
from compiler.validation import SEVERITY_ERROR

pytestmark = pytest.mark.unit

def test_purity_validator_success():
    """Verify that a graph with serializable params passes."""
    graph = Graph()
    graph.add_node("node1", "TestOp", params={
        "learning_rate": 0.01,
        "model_handle": "q_net",
        "buffer_id": "main",
        "dims": [64, 128]
    })
    
    report = validate_ir_purity(graph)
    assert not report.has_errors()
    assert len(report.issues) == 0

def test_purity_validator_p001_fail_module():
    """Verify failure for P001 (nn.Module)."""
    graph = Graph()
    model = nn.Linear(10, 10)
    graph.add_node("node1", "TestOp", params={"model": model})
    
    report = validate_ir_purity(graph)
    assert report.has_errors()
    assert any(i.code == "P001" for i in report.issues)
    assert any("torch.nn.Module" in i.message for i in report.issues)

def test_purity_validator_p002_fail_optimizer():
    """Verify failure for P002 (Optimizer)."""
    graph = Graph()
    params = [torch.nn.Parameter(torch.zeros(1))]
    optimizer = optim.Adam(params)
    graph.add_node("node1", "TestOp", params={"opt": optimizer})
    
    report = validate_ir_purity(graph)
    assert report.has_errors()
    assert any(i.code == "P002" for i in report.issues)

def test_purity_validator_p003_fail_buffer():
    """Verify failure for P003 (ReplayBuffer)."""
    graph = Graph()
    rb = ReplayBuffer(capacity=100)
    graph.add_node("node1", "TestOp", params={"buffer": rb})
    
    report = validate_ir_purity(graph)
    assert report.has_errors()
    assert any(i.code == "P003" for i in report.issues)

def test_purity_validator_p004_fail_callable():
    """Verify failure for P004 (Callable closures)."""
    graph = Graph()
    
    # Lambda
    graph.add_node("node1", "TestOp", params={"func": lambda x: x + 1})
    
    # Closure
    def my_closure(x):
        return x * 2
    graph.add_node("node2", "TestOp", params={"func": my_closure})
    
    report = validate_ir_purity(graph)
    assert report.has_errors()
    assert len(report.issues) == 2
    assert all(i.code == "P004" for i in report.issues)

def test_purity_validator_multiple_failures():
    """Verify that multiple violations are caught in a single pass."""
    graph = Graph()
    model = nn.Linear(10, 10)
    rb = ReplayBuffer(capacity=100)
    
    graph.add_node("node1", "TestOp", params={
        "model": model,
        "buffer": rb,
        "valid_param": 123
    })
    
    report = validate_ir_purity(graph)
    assert len(report.issues) == 2
    codes = {i.code for i in report.issues}
    assert codes == {"P001", "P003"}
