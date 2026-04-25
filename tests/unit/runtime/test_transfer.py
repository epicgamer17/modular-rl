import pytest
import torch
import gymnasium as gym
from core.graph import Graph, NODE_TYPE_SOURCE
from runtime.executor import execute
from runtime.refs import DataRef, StorageLocation
from runtime.state import ReplayBuffer

pytestmark = pytest.mark.unit


def test_explicit_transfer_semantics():
    """
    Test 10.2: Verify explicit transfer nodes under a complex device placement.
    Replay (CPU) -> Transfer -> Learner (GPU)
    """
    # 1. Setup Graph
    graph = Graph()
    graph.add_node("replay_out", NODE_TYPE_SOURCE)

    # Semantic Transfer Node
    graph.add_node("to_gpu", "TransferToDevice", params={"device_id": 0})
    graph.add_edge("replay_out", "to_gpu", dst_port="input")

    # Mock Learner that REQUIRES GPU data
    def op_learner(node, inputs, context=None):
        data_ref = inputs["input"]
        # ASSERT: No silent copy. Data must ALREADY be on GPU.
        assert data_ref.location == StorageLocation.GPU
        assert data_ref.data.is_cuda if torch.cuda.is_available() else True
        return "update_done"

    from runtime.operator_registry import register_operator

    register_operator("Learner", op_learner)

    graph.add_node("update", "Learner")
    graph.add_edge("to_gpu", "update", dst_port="input")

    # 2. Execution with CPU-originated data
    cpu_data = torch.randn(10)
    cpu_ref = DataRef(cpu_data, location=StorageLocation.CPU)

    results = execute(graph, initial_inputs={"replay_out": cpu_ref})

    # 3. Verification
    assert results["update"].data == "update_done"

    # Verify the transfer history of the object that reached the learner
    to_gpu_ref = results["to_gpu"]
    assert to_gpu_ref.location == StorageLocation.GPU
    assert any(h["reason"] == "explicit_move" for h in to_gpu_ref.transfer_history)


def test_serialization_semantics():
    """Verify serialize/deserialize nodes for remote worker simulation."""
    graph = Graph()
    graph.add_node("data_in", NODE_TYPE_SOURCE)
    graph.add_node("serialize", "Serialize")
    graph.add_node("deserialize", "Deserialize")

    graph.add_edge("data_in", "serialize", dst_port="input")
    graph.add_edge("serialize", "deserialize", dst_port="input")

    data = torch.tensor([1, 2, 3])
    results = execute(graph, initial_inputs={"data_in": data})

    assert isinstance(results["serialize"].data, bytes)
    assert isinstance(results["deserialize"], DataRef)
    assert torch.equal(results["deserialize"].data, data)
    assert results["deserialize"].location == StorageLocation.CPU


if __name__ == "__main__":
    test_explicit_transfer_semantics()
    test_serialization_semantics()
