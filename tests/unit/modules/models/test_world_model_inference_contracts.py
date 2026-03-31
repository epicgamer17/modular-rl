import pytest
import torch
from torch import nn
from modules.models.world_model import WorldModel
from modules.heads.base import BaseHead, HeadOutput
from agents.learner.losses.representations import ClassificationRepresentation

pytestmark = pytest.mark.unit

class MockHead(BaseHead):
    def __init__(self, name, training_val, inference_val, **kwargs):
        # We need a representation for BaseHead
        from agents.learner.losses.representations import ScalarRepresentation
        rep = ScalarRepresentation()
        super().__init__(input_shape=(64,), representation=rep, name=name)
        self.training_val = training_val
        self.inference_val = inference_val

    def forward(self, x, **kwargs):
        # Return tensors with same batch size as x
        B = x.shape[0]
        return HeadOutput(
            training_tensor=torch.full((B,), self.training_val, device=x.device),
            inference_tensor=torch.full((B,), self.inference_val, device=x.device),
            state={},
            metrics={}
        )

def test_world_model_recurrent_inference_contract():
    """
    Tier 1 Unit Test: WorldModel recurrent_inference Contract
    - Verify that recurrent_inference correctly populates WorldModelOutput fields.
    - Verify that it uses the correct keys from head outputs.
    - Verify fail-fast behavior for missing critical heads.
    """
    hidden_dim = 64
    num_actions = 4
    B = 2

    # 1. Setup Mock Heads
    def make_reward_head(**kwargs):
        return MockHead(name="reward_logits", training_val=1.1, inference_val=2.2)
    
    def make_to_play_head(**kwargs):
        return MockHead(name="to_play_logits", training_val=3.3, inference_val=0) # player 0

    env_head_fns = {
        "reward_logits": make_reward_head,
        "to_play_logits": make_to_play_head,
    }

    class MockDynamics(nn.Module):
        def __init__(self, input_shape):
            super().__init__()
            self.output_shape = input_shape
        def forward(self, x):
            return x

    model = WorldModel(
        latent_dimensions=(hidden_dim,),
        num_actions=num_actions,
        stochastic=False,
        dynamics_fn=MockDynamics,
        env_head_fns=env_head_fns
    )

    latent = torch.randn((B, hidden_dim))
    action = torch.randint(0, num_actions, (B, 1))

    # 2. Run Inference
    out = model.recurrent_inference(latent, action)

    # 3. Assertions
    # reward (logits) should be 1.1
    assert torch.allclose(out.reward, torch.tensor(1.1)), f"Expected reward logits 1.1, got {out.reward}"
    # instant_reward (scalar) should be 2.2
    assert torch.allclose(out.instant_reward, torch.tensor(2.2)), f"Expected instant_reward 2.2, got {out.instant_reward}"
    # to_play_logits should be 3.3
    assert torch.allclose(out.to_play_logits, torch.tensor(3.3)), f"Expected to_play_logits 3.3, got {out.to_play_logits}"
    # to_play (index) should be 0
    assert (out.to_play == 0).all()

    # 4. Continuation should be None (not provided in mock)
    assert out.continuation is None
    assert out.continuation_logits is None

def test_world_model_recurrent_inference_fail_fast():
    """Verify that recurrent_inference fails fast when a required head is missing."""
    hidden_dim = 64
    num_actions = 4
    B = 1

    # Missing 'reward_logits' head
    env_head_fns = {
        "to_play_logits": lambda **kwargs: MockHead(name="to_play_logits", training_val=0, inference_val=0),
    }

    class MockDynamics(nn.Module):
        def __init__(self, input_shape):
            super().__init__()
            self.output_shape = input_shape
        def forward(self, x):
            return x

    model = WorldModel(
        latent_dimensions=(hidden_dim,),
        num_actions=num_actions,
        stochastic=False,
        dynamics_fn=MockDynamics,
        env_head_fns=env_head_fns
    )

    latent = torch.randn((B, hidden_dim))
    action = torch.randint(0, num_actions, (B, 1))

    with pytest.raises(KeyError) as excinfo:
        model.recurrent_inference(latent, action)
    
    assert "reward_logits" in str(excinfo.value)
