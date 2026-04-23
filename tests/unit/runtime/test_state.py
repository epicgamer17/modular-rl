import pytest
import torch
from runtime.state import ReplayBuffer, ParameterStore, OptimizerState

pytestmark = pytest.mark.unit

def test_replay_buffer_operations():
    """Verify ReplayBuffer insertion and deterministic sampling."""
    rb = ReplayBuffer(capacity=5)
    
    # 1. Insert transitions
    for i in range(5):
        rb.add({
            "obs": torch.tensor([float(i)]),
            "action": torch.tensor([i])
        })
    
    assert len(rb) == 5
    
    # 2. Circularity check
    rb.add({"obs": torch.tensor([99.0]), "action": torch.tensor([99])})
    assert len(rb) == 5
    # The first element (index 0) should have been overwritten
    # Note: sample() doesn't guarantee order, but we can check the internal buffer
    assert rb.buffer[0]["obs"].item() == 99.0

    # 3. Deterministic sampling
    batch1 = rb.sample(batch_size=2, seed=42)
    batch2 = rb.sample(batch_size=2, seed=42)
    
    assert len(batch1) == 2
    for t1, t2 in zip(batch1, batch2):
        assert torch.equal(t1["obs"], t2["obs"])
        assert torch.equal(t1["action"], t2["action"])

def test_parameter_store_updates():
    """Verify ParameterStore versioning and in-place updates."""
    initial_w = torch.zeros(2, 2)
    store = ParameterStore({"weight": initial_w})
    
    assert store.version == 0
    
    # Update
    new_w = torch.ones(2, 2)
    store.update_state({"weight": new_w})
    
    assert store.version == 1
    assert torch.all(store.get_state()["weight"] == 1.0)
    # Check in-place mutation vs replacement
    # Our implementation uses .copy_() for existing keys
    assert initial_w.sum() == 4.0 # initial_w was modified in-place

def test_optimizer_state_management():
    """Verify OptimizerState can wrap and step an optimizer."""
    params = [torch.nn.Parameter(torch.randn(2, 2))]
    opt = torch.optim.SGD(params, lr=0.1)
    state = OptimizerState(opt)
    
    initial_val = params[0].clone()
    
    # Mock loss and step
    loss = (params[0] ** 2).sum()
    state.step(loss)
    
    assert not torch.equal(params[0], initial_val), "Parameters should have updated after step."

if __name__ == "__main__":
    test_replay_buffer_operations()
    test_parameter_store_updates()
    test_optimizer_state_management()
    print("Test 3.3 Passed!")
