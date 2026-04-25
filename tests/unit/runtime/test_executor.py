import pytest
import torch
from core.graph import Graph, NODE_TYPE_SOURCE
from runtime.executor import execute, register_operator

pytestmark = pytest.mark.unit

def test_minimal_linear_graph():
    """
    Test 3.1: Verify execution of a trivial Linear graph.
    Obs (Source) -> Linear -> Output
    """
    # 1. Define and register operators
    from runtime.signals import NoOp
    def run_source(node, inputs, context=None):
        # Source nodes just pass through initial values provided in execute()
        return NoOp()
        
    def run_linear(node, inputs, context=None):
        # Simple linear transformation: y = x @ W
        # Expecting 'input' port
        x = inputs["input"]
        w = node.params["weight"]
        return x @ w
    
    register_operator(NODE_TYPE_SOURCE, run_source)
    register_operator("Linear", run_linear)
    
    # 2. Construct Graph
    graph = Graph()
    graph.add_node("obs", NODE_TYPE_SOURCE)
    
    # Weight matrix [2, 2]
    W = torch.tensor([[1.0, 2.0], 
                      [3.0, 4.0]])
    graph.add_node("layer1", "Linear", params={"weight": W})
    
    graph.add_edge("obs", "layer1", dst_port="input")
    
    # 3. Execution
    # Input vector [1, 1]
    x = torch.tensor([1.0, 1.0])
    results = execute(graph, initial_inputs={"obs": x})
    
    # 4. Verification
    # Result should be [1*1 + 1*3, 1*2 + 1*4] = [4, 6]
    expected = torch.tensor([4.0, 6.0])
    
    assert "layer1" in results
    assert torch.allclose(results["layer1"].data, expected)
    print("\nExecution Results:")
    print(f"  obs: {results['obs']}")
    print(f"  layer1: {results['layer1']}")

if __name__ == "__main__":
    test_minimal_linear_graph()
    print("Test 3.1 Passed!")
