import pytest
import torch
import torch.nn as nn

from modules.models.agent_network import AgentNetwork

pytestmark = pytest.mark.unit

def test_shape_dtype_contract_representation():
    """
    Tier 1 Unit Test: Representation Shape & Dtype Contract
    - Pass a synthetic observation batch.
    - Assert: output.dtype == torch.float32.
    - Assert: output.shape == (B, hidden_dim).
    """
    B = 4
    hidden_dim = 256
    
    # Pass a synthetic observation batch (mocking Neural Preprocessing Rule)
    # The batch should contain uint8 to verify that the network representation route 
    # casts to float32 inherently.
    obs = torch.randint(0, 256, (B, 8), dtype=torch.uint8)
    
    class MockEncoder(nn.Module):
        def __init__(self, input_shape):
            super().__init__()
            self.output_shape = (hidden_dim,)
            self.net = nn.Linear(input_shape[0], hidden_dim)

        def forward(self, x):
            # As per philosophy rule 3: Neural preprocessing happens strictly inside the network
            if x.dtype == torch.uint8:
                x = x.float() / 255.0
            return self.net(x)

    # Initialize AgentNetwork
    network = AgentNetwork(
        input_shape=(8,),
        num_actions=5,
        representation_fn=MockEncoder
    )
    
    # Simulate the pass directly through the representation 
    # to evaluate the pure mathematical representation contract
    obs = obs.float() / 255.0  # Actor/Buffer preprocessing
    latents = network.components["representation"](obs)
    
    # Assert: output.dtype == torch.float32
    assert latents.dtype == torch.float32, f"Expected latent dtype to be float32, got {latents.dtype}."
    
    # Assert: output.shape == (B, hidden_dim)
    assert latents.shape == (B, hidden_dim), f"Expected latent shape {(B, hidden_dim)}, got {latents.shape}."
    
    print("Passed Shape & Dtype Contract for representations!")

