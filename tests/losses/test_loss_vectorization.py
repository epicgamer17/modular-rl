import pytest
import torch
import torch.nn.functional as F
import numpy as np
from agents.learner.losses.losses import (
    LossPipeline, ValueLoss, PolicyLoss, RewardLoss, 
    StandardDQNLoss, C51Loss, ConsistencyLoss,
    ToPlayLoss, RelativeToPlayLoss
)

pytestmark = pytest.mark.unit

@pytest.fixture
def mock_agent_network():
    class MockNetwork(torch.nn.Module):
        def project(self, x, grad=True):
            # Identity projection for testing
            return x 
    return MockNetwork()

def test_muzero_losses_vectorized(muzero_config, mock_agent_network):
    """
    Verify that MuZero losses (Value, Policy, Reward, Consistency) 
    work correctly with vectorized [B, T] inputs.
    """
    device = torch.device("cpu")
    torch.manual_seed(42)
    
    # 1. Setup Modules
    v_loss = ValueLoss(muzero_config, device)
    p_loss = PolicyLoss(muzero_config, device)
    r_loss = RewardLoss(muzero_config, device)
    c_loss = ConsistencyLoss(muzero_config, device, mock_agent_network)
    
    pipeline = LossPipeline([v_loss, p_loss, r_loss, c_loss])
    
    B, T = 2, 4  # T = unroll_steps (3) + 1
    num_actions = 2 # CartPole actions
    latent_dim = 8
    
    # 2. Create Vectorized Data [B, T, ...]
    predictions = {
        "values": torch.randn(B, T, device=device),
        "policies": torch.randn(B, T, num_actions, device=device),
        "rewards": torch.randn(B, T, device=device),
        "latents": torch.randn(B, T, latent_dim, device=device),
    }
    
    targets = {
        "values": torch.randn(B, T, device=device),
        "policies": F.softmax(torch.randn(B, T, num_actions, device=device), dim=-1),
        "rewards": torch.randn(B, T, device=device),
        "consistency_targets": torch.randn(B, T, latent_dim, device=device),
        "value_mask": torch.ones(B, T, dtype=torch.bool, device=device),
        "policy_mask": torch.ones(B, T, dtype=torch.bool, device=device),
        "reward_mask": torch.ones(B, T, dtype=torch.bool, device=device),
    }
    
    # 3. Run Pipeline
    scales = torch.ones(1, T, device=device) * 0.5
    total_losses, logs, priorities = pipeline.run(
        predictions, targets, gradient_scales=scales
    )
    
    # 4. Verify Correctness
    assert "default" in total_losses, "Missing default optimizer loss"
    assert total_losses["default"].ndim == 0, "Total loss should be a scalar"
    assert not torch.isnan(total_losses["default"]), "Loss is NaN"
    
    for name in ["ValueLoss", "PolicyLoss", "RewardLoss", "ConsistencyLoss"]:
        assert name in logs, f"Missing {name} in log dict"
        assert isinstance(logs[name], float), f"{name} log should be float"

def test_dqn_losses_vectorized(rainbow_config):
    """
    Verify that DQN losses (StandardDQN, C51) work with vectorized [B, T] inputs.
    Even if T=1 normally, our code must handle any T.
    """
    device = torch.device("cpu")
    torch.manual_seed(42)
    
    # Standard DQN
    dqn_loss = StandardDQNLoss(rainbow_config, device)
    
    B, T = 2, 1
    num_actions = 2 # From rainbow_config (CartPole)
    
    predictions = {
        "q_values": torch.randn(B, T, num_actions, device=device)
    }
    targets = {
        "q_values": torch.randn(B, T, device=device),
        "actions": torch.zeros(B, T, dtype=torch.long, device=device),
        "value_mask": torch.ones(B, T, dtype=torch.bool, device=device)
    }
    
    pipeline = LossPipeline([dqn_loss])
    total_losses, logs, priorities = pipeline.run(predictions, targets)
    
    assert "StandardDQNLoss" in logs
    assert priorities.shape == (B,)

def test_to_play_losses_vectorized(muzero_config):
    """
    Verify ToPlayLoss and RelativeToPlayLoss vectorized logic.
    """
    device = torch.device("cpu")
    torch.manual_seed(42)
    
    # Set num_players > 1 for these losses
    muzero_config.game.num_players = 2
    
    tp_loss = ToPlayLoss(muzero_config, device)
    rtp_loss = RelativeToPlayLoss(muzero_config, device)
    
    pipeline = LossPipeline([tp_loss, rtp_loss])
    
    B, T = 2, 4
    num_players = 2
    
    predictions = {
        "to_plays": torch.randn(B, T, num_players, device=device)
    }
    targets = {
        "to_plays": torch.tensor([[0, 1, 0, 1], [0, 1, 1, 0]], device=device),
        "to_play_mask": torch.ones(B, T, dtype=torch.bool, device=device)
    }
    
    total_losses, logs, priorities = pipeline.run(predictions, targets)
    
    assert "ToPlayLoss" in logs
    assert "RelativeToPlayLoss" in logs
    assert not torch.isnan(total_losses["default"])

def test_ppo_losses_vectorized(ppo_config):
    """
    Verify PPO losses (Policy, Value) vectorized logic.
    """
    device = torch.device("cpu")
    torch.manual_seed(42)
    
    from agents.learner.losses.losses import PPOPolicyLoss, PPOValueLoss
    
    # PPO typically has clip_param and entropy_coeff set
    pol_loss = PPOPolicyLoss(ppo_config, device, clip_param=0.2, entropy_coefficient=0.01)
    val_loss = PPOValueLoss(ppo_config, device, critic_coefficient=0.5)
    
    pipeline = LossPipeline([pol_loss, val_loss])
    
    B, T = 2, 1
    num_actions = 2 # From ppo_config (CartPole)
    
    predictions = {
        "policies": torch.randn(B, T, num_actions, device=device),
        "values": torch.randn(B, T, device=device)
    }
    targets = {
        "actions": torch.zeros(B, T, dtype=torch.long, device=device),
        "old_log_probs": torch.randn(B, T, device=device),
        "advantages": torch.randn(B, T, device=device),
        "returns": torch.randn(B, T, device=device),
        "policy_mask": torch.ones(B, T, dtype=torch.bool, device=device),
        "value_mask": torch.ones(B, T, dtype=torch.bool, device=device),
    }
    
    total_losses, logs, priorities = pipeline.run(predictions, targets)
    
    assert "PPOPolicyLoss" in logs
    assert "PPOValueLoss" in logs
    assert "approx_kl" in logs
