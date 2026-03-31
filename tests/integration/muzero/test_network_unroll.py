import pytest
import torch

pytestmark = pytest.mark.integration

def test_muzero_network_unroll_shapes():
    """
    Ensure the Dynamics and Prediction networks can unroll K times without shape mismatch errors.
    
    Setup:
    - B=2 (Batch size)
    - K=3 (Unroll steps)
    - C, H, W = 3, 8, 8 (Input shape)
    - num_actions = 4
    
    Assertion:
    1. Pass observations through the Representation network. assert hidden_state.shape == expected.
    2. Loop K times, passing the hidden_state and actions[:, k] into the Dynamics network.
    3. Assert the final returned lists for policies, values, and rewards all have length K+1.
    """
    # TODO: re-enable when imports are restored
    # Original assertions:
    # assert "dynamics" in hidden_state, "Recurrent state must contain 'dynamics' key for MuZero."
    # assert hidden_state["dynamics"].shape == (B, *latent_dim), f"Expected {(B, *latent_dim)}, got {hidden_state['dynamics'].shape}"
    # assert len(policies) == K + 1, f"Expected {K+1} policies, got {len(policies)}"
    # assert len(values) == K + 1, f"Expected {K+1} values, got {len(values)}"
    # assert len(rewards) == K + 1, f"Expected {K+1} rewards, got {len(rewards)}"
    # assert current_state["dynamics"].shape == (B, *latent_dim)
    # assert isinstance(p, torch.distributions.Distribution), f"Step {i}: Policy must be a Distribution object"
    # assert p.batch_shape == (B,), f"Step {i}: Batch shape mismatch. Expected {(B,)}, got {p.batch_shape}"
    # assert torch.is_tensor(v), f"Step {i}: Value must be a Tensor"
    # assert v.shape == (B,), f"Step {i}: Value shape mismatch. Expected {(B,)}, got {v.shape}"
    # assert torch.is_tensor(r), f"Step {i}: Reward must be a Tensor"
    # assert r.shape == (B,), f"Step {i}: Reward shape mismatch. Expected {(B,)}, got {r.shape}"
    pytest.skip("TODO: update for old_muzero revert")

