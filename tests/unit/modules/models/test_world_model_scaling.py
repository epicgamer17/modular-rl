import pytest
import torch
import torch.nn as nn
from modules.models.world_model import WorldModel

pytestmark = pytest.mark.unit

def test_dynamics_hidden_state_scaling():
    """
    Tier 1 Unit Test: Dynamics Hidden State Normalization
    - Pass a hidden state and an action through the dynamics function g
    - Assert: min(s) = 0.0 and max(s) = 1.0 to ensure the min-max normalization bounds the activations correctly.
    """
    B, hidden_dim = 2, 64
    num_actions = 4
    
    # A simple MLP dynamics backbone that outputs values far outside [0, 1]
    class MockDynamics(nn.Module):
        def __init__(self, input_shape):
            super().__init__()
            self.output_shape = input_shape
            self.net = nn.Linear(input_shape[0], input_shape[0])
            # Force weights to initially output large/negative values
            nn.init.uniform_(self.net.weight, a=-10.0, b=10.0)
            
        def forward(self, x):
            return self.net(x)

    from modules.heads.base import BaseHead, HeadOutput

    class MockHead(BaseHead):
        def __init__(self, **kwargs):
            from agents.learner.losses.representations import ScalarRepresentation
            super().__init__(input_shape=(64,), representation=ScalarRepresentation(), name="reward_logits")
        def forward(self, x, **kwargs):
            return HeadOutput(
                training_tensor=torch.zeros_like(x[:, 0]),
                inference_tensor=torch.zeros_like(x[:, 0]),
                state={}, metrics={}
            )

    model = WorldModel(
        latent_dimensions=(hidden_dim,),
        num_actions=num_actions,
        stochastic=False,
        dynamics_fn=MockDynamics,
        action_embedding_dim=16,
        env_head_fns={"reward_logits": lambda **kwargs: MockHead()}
    )
    
    latent = torch.randn((B, hidden_dim)) * 10.0
    action = torch.randint(0, num_actions, (B, 1))
    
    # Run through the dynamics step 
    out = model.recurrent_inference(latent, action)
    next_latent = out.features
    
    # Check scaling across the feature dimension
    for i in range(B):
        # min-max normalization forces exactly 0.0 and 1.0 (with slight tolerance for float math)
        assert torch.isclose(next_latent[i].min(), torch.tensor(0.0), atol=1e-5), f"Min value is {next_latent[i].min()}"
        assert torch.isclose(next_latent[i].max(), torch.tensor(1.0), atol=1e-5), f"Max value is {next_latent[i].max()}"
