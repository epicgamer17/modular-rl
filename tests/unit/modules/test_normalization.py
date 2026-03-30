import torch
import torch.nn as nn
import pytest
from modules.utils import _normalize_hidden_state
from modules.models.world_model import DeterministicDynamics, WorldModel
from modules.models.agent_network import AgentNetwork
from modules.heads.base import HeadOutput

pytestmark = pytest.mark.unit

def test_normalization_utils():
    """Verify that _normalize_hidden_state correctly scales to [0,1]."""
    # 1. Test Dense
    S_dense = torch.tensor([[10.0, 20.0, 30.0], [0.0, 5.0, 10.0]]) # B=2, W=3
    norm_dense = _normalize_hidden_state(S_dense)
    assert norm_dense.min() >= 0.0
    assert norm_dense.max() <= 1.0
    assert torch.allclose(norm_dense[0], torch.tensor([0.0, 0.5, 1.0]))
    
    # 2. Test Spatial (B, C, H, W)
    S_spatial = torch.randn(2, 4, 8, 8) * 100.0
    norm_spatial = _normalize_hidden_state(S_spatial)
    assert norm_spatial.min() >= -1e-6
    assert norm_spatial.max() <= 1.0 + 1e-6
    
    for b in range(2):
        assert torch.allclose(norm_spatial[b].min(), torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(norm_spatial[b].max(), torch.tensor(1.0), atol=1e-5)

class MockHead(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.last_input = None
        self.input_source = "default"
    def forward(self, x, **kwargs):
        self.last_input = x.clone()
        return HeadOutput(
            training_tensor=torch.zeros_like(x), 
            inference_tensor=torch.zeros_like(x),
            state={},
            metrics={}
        )

class MockDynamics(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.output_shape = (16,)
        self.last_input = None
    def forward(self, x):
        self.last_input = x.clone()
        return x

def test_muzero_dynamics_input_normalized():
    """Verify that the input to the dynamics backbone is normalized."""
    latent_dim = (16,)
    dynamics_backbone = MockDynamics()
    
    # Dynamics pipeline (Deterministic)
    dw = DeterministicDynamics(
        latent_dimensions=latent_dim,
        num_actions=2,
        dynamics_fn=lambda **kwargs: dynamics_backbone,
        action_embedding_dim=8
    )
    
    # Pass a HUGE unnormalized latent
    unnorm_latent = torch.randn(1, 16) * 100.0
    norm_latent = _normalize_hidden_state(unnorm_latent)
    
    action = torch.zeros(1, dtype=torch.long)
    
    # Step. Input to DW.forward() should be normalized.
    dw.forward(norm_latent, action)
    
    # The input to the backbone (MockDynamics) is current_latent + action_emb.
    # We want to verify that the 'current_latent' part was the normalized one.
    # Since DynamicsFusion is: state + action_emb.
    # We can check that the values are not 'huge'.
    assert dynamics_backbone.last_input.abs().max() < 10.0 # Normalized part [0,1] + emb
    
def test_muzero_unnormalized_reward_input():
    """Verify that reward heads receive unnormalized hidden states."""
    latent_dim = (16,)
    dynamics_backbone = MockDynamics()
    reward_head = MockHead()
    policy_head = MockHead()
    
    env_heads = {
        "reward_logits": lambda **kwargs: reward_head, 
        "policy_logits": lambda **kwargs: policy_head
    }
    
    wm = WorldModel(
        latent_dimensions=latent_dim,
        num_actions=2,
        dynamics_fn=lambda **kwargs: dynamics_backbone,
        env_head_fns=env_heads
    )
    
    h = torch.randn(1, 16) * 10.0
    initial_unnorm = h.clone()
    initial_norm = _normalize_hidden_state(initial_unnorm)
    
    # Run recurrent inference
    out = wm.recurrent_inference(initial_norm, torch.zeros(1, dtype=torch.long))
    
    # Dynamics was called once.
    # Predicted r1 from (s0, a0).
    # reward_head.last_input should be unnormalized r1?
    # Wait! In recurrent_inference:
    # 1. dynamics(s0, a0) -> (s1_norm, s1_unnorm)
    # 2. reward_head(s1_unnorm)
    # So reward_head.last_input should be s1_unnorm.
    
    # Verify that it is NOT the normalized one
    assert not torch.allclose(reward_head.last_input, out.features)
    # Verify policy_head used the normalized one
    assert torch.allclose(policy_head.last_input, out.features)

def test_agent_network_representation_normalization():
    """Verify that AgentNetwork normalizes latent features from representation."""
    input_shape = (4,)
    # Use a backbone that returns huge values
    representation_fn = lambda **kwargs: nn.Sequential(nn.Linear(4, 16), nn.ConstantPad1d((0, 0), 1000.0))
    head_fns = {"policy_logits": lambda **kwargs: MockHead()}
    
    agent = AgentNetwork(
        input_shape=input_shape,
        num_actions=2,
        representation_fn=representation_fn,
        head_fns=head_fns
    )
    
    obs = torch.randn(1, 4)
    # 1. Check obs_inference
    out = agent.obs_inference(obs)
    # The latent stored in recurrent_state['dynamics'] should be normalized
    latent = out.recurrent_state["dynamics"]
    assert latent.min() >= 0.0
    assert latent.max() <= 1.0
    
    # 2. Check learner_inference (unrolls)
    batch = {
        "observations": torch.randn(1, 1, 4), # B, T, D
        "actions": torch.zeros((1, 1), dtype=torch.long)
    }
    # If no world model, 'latents' in final_output should be normalized
    out_train = agent.learner_inference(batch)
    latents = out_train["latents"] # [B, T, D]
    assert latents.min() >= 0.0
    assert latents.max() <= 1.0

def test_muzero_dynamics_output_normalized():
    """Verify that WorldModel dynamics produces normalized latent outputs."""
    latent_dim = (16,)
    # Backdrop that returns raw values
    dynamics_fn = lambda **kwargs: nn.Linear(16, 16)
    
    wm = WorldModel(
        latent_dimensions=latent_dim,
        num_actions=2,
        dynamics_fn=dynamics_fn,
        env_head_fns={}
    )
    
    # Step dynamics with normalized input
    h_norm = torch.rand(1, 16) # [0, 1]
    res = wm.dynamics_pipeline(h_norm, torch.zeros(1, dtype=torch.long))
    
    # next_latent must be normalized
    assert res["next_latent"].min() >= 0.0
    assert res["next_latent"].max() <= 1.0
    
def test_ppo_no_representation():
    """Verify that AgentNetwork allows no representation (Identity)."""
    input_shape = (4,)
    head_fns = {"policy_logits": lambda **kwargs: MockHead()}
    
    agent = AgentNetwork(
        input_shape=input_shape,
        num_actions=2,
        representation_fn=None,
        head_fns=head_fns
    )
    
    obs = torch.randn(1, 4)
    out = agent.obs_inference(obs)
    
    assert "world_model" not in agent.components
    assert out.policy is not None

if __name__ == "__main__":
    # Workaround for environmental pytest permission issues
    print("\n" + "="*50)
    print("RUNNING NORMALIZATION TESTS (STANDALONE)")
    print("="*50)
    
    test_funcs = [
        test_normalization_utils,
        test_muzero_dynamics_input_normalized,
        test_muzero_dynamics_output_normalized,
        test_muzero_unnormalized_reward_input,
        test_agent_network_representation_normalization,
        test_ppo_no_representation
    ]
    
    passed = 0
    for func in test_funcs:
        try:
            print(f"RUNNING: {func.__name__}...", end=" ", flush=True)
            func()
            print("PASSED")
            passed += 1
        except Exception as e:
            print(f"FAILED\nERROR: {e}")
            import traceback
            traceback.print_exc()
            
    print("="*50)
    print(f"RESULTS: {passed}/{len(test_funcs)} PASSED")
    print("="*50 + "\n")
    
    if passed < len(test_funcs):
        import sys
        sys.exit(1)
