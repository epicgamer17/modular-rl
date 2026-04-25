import pytest
import torch
from core.graph import Node
from core.batch import TransitionBatch
from ops.rl.advantage import op_advantage_estimation, ADVANTAGE_KERNELS

pytestmark = pytest.mark.unit

def test_advantage_dispatch_correctness():
    """Verify that op_advantage_estimation calls the correct kernel."""
    T, N = 5, 2
    batch = TransitionBatch(
        obs=torch.randn(T, N, 4),
        action=torch.zeros(T, N),
        reward=torch.randn(T, N),
        next_obs=torch.randn(T, N, 4),
        done=torch.zeros(T, N),
        value=torch.randn(T, N),
    )
    next_value = torch.randn(N)
    next_terminated = torch.zeros(N)
    gamma = 0.99
    
    # Test GAE dispatch
    node = Node(node_id="adv", node_type="AdvantageEstimation", params={"method": "gae", "gamma": gamma, "gae_lambda": 0.95})
    inputs = {"batch": batch, "next_value": next_value, "next_terminated": next_terminated}
    
    output = op_advantage_estimation(node, inputs)
    
    # Manually call kernel to compare
    expected_adv, expected_ret = ADVANTAGE_KERNELS["gae"](batch, next_value, next_terminated, gamma, gae_lambda=0.95)
    
    assert torch.allclose(output["advantages"], expected_adv)
    assert torch.allclose(output["returns"], expected_ret)

def test_advantage_unknown_method_rejected():
    """Verify that unknown methods raise ValueError."""
    T, N = 5, 2
    batch = TransitionBatch(
        obs=torch.randn(T, N, 4),
        action=torch.zeros(T, N),
        reward=torch.randn(T, N),
        next_obs=torch.randn(T, N, 4),
        done=torch.zeros(T, N),
        value=torch.randn(T, N),
    )
    next_value = torch.randn(N)
    next_terminated = torch.zeros(N)
    
    node = Node(node_id="adv", node_type="AdvantageEstimation", params={"method": "unknown_method", "gamma": 0.99})
    inputs = {"batch": batch, "next_value": next_value, "next_terminated": next_terminated}
    
    with pytest.raises(ValueError, match="Unknown advantage estimation method: unknown_method"):
        op_advantage_estimation(node, inputs)

def test_advantage_flattened_dispatch():
    """Verify that flattened batches are correctly handled and dispatched."""
    T, N = 5, 2
    batch = TransitionBatch(
        obs=torch.randn(T*N, 4),
        action=torch.zeros(T*N),
        reward=torch.randn(T*N),
        next_obs=torch.randn(T*N, 4),
        done=torch.zeros(T*N),
        value=torch.randn(T*N),
        terminated=torch.zeros(T*N)
    )
    next_value = torch.randn(N)
    next_terminated = torch.zeros(N)
    gamma = 0.99
    
    node = Node(node_id="adv", node_type="AdvantageEstimation", params={"method": "gae", "gamma": gamma, "num_envs": N})
    inputs = {"batch": batch, "next_value": next_value, "next_terminated": next_terminated}
    
    output = op_advantage_estimation(node, inputs)
    
    assert output["advantages"].shape == (T*N,)
    assert output["returns"].shape == (T*N,)
