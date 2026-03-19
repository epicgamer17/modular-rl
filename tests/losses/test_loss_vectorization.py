import pytest
import torch
import torch.nn.functional as F
import numpy as np
from agents.learner.losses.losses import (
    LossPipeline, ValueLoss, PolicyLoss, RewardLoss, 
    StandardDQNLoss, C51Loss, ConsistencyLoss,
    ToPlayLoss, RelativeToPlayLoss
)
from agents.learner.losses.representations import (
    ScalarRepresentation, ClassificationRepresentation, 
    IdentityRepresentation, TwoHotRepresentation, CategoricalRepresentation
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
    
    # Configure muozero_config to expect distributional shapes
    muzero_config.support_range = 300
    muzero_config.atom_size = 601
    
    # 1. Setup Modules with Mandatory Representations
    # MuZero typically uses TwoHot for values and rewards
    v_rep = TwoHotRepresentation(-300, 300, 601)
    p_rep = ClassificationRepresentation(2) # 2 actions
    r_rep = TwoHotRepresentation(-300, 300, 601)
    
    v_loss = ValueLoss(muzero_config, device, representation=v_rep)
    p_loss = PolicyLoss(muzero_config, device, representation=p_rep)
    r_loss = RewardLoss(muzero_config, device, representation=r_rep)
    c_loss = ConsistencyLoss(muzero_config, device, representation=IdentityRepresentation(), agent_network=mock_agent_network)
    
    pipeline = LossPipeline([v_loss, p_loss, r_loss, c_loss])
    
    B, T = 2, 4  
    num_actions = 2 
    latent_dim = 8
    
    # 2. Create Vectorized Data [B, T, ...]
    # Values/Rewards are TwoHot (logits)
    predictions = {
        "values": torch.randn(B, T, 601, device=device),
        "policies": torch.randn(B, T, num_actions, device=device),
        "rewards": torch.randn(B, T, 601, device=device),
        "latents": torch.randn(B, T, latent_dim, device=device),
    }
    
    targets = {
        "values": torch.randn(B, T, device=device), # TwoHot will project this
        "policies": F.softmax(torch.randn(B, T, num_actions, device=device), dim=-1),
        "rewards": torch.randn(B, T, device=device), # TwoHot will project this
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
    assert "default" in total_losses
    assert total_losses["default"].ndim == 0
    
    for name in ["ValueLoss", "PolicyLoss", "RewardLoss", "ConsistencyLoss"]:
        assert name in logs

def test_dqn_losses_vectorized(rainbow_config):
    """
    Verify that DQN losses work with vectorized [B, T] inputs.
    """
    device = torch.device("cpu")
    torch.manual_seed(42)
    
    # Standard DQN uses ScalarRepresentation
    dqn_loss = StandardDQNLoss(rainbow_config, device, representation=ScalarRepresentation())
    
    B, T = 2, 1
    num_actions = 2 
    
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

def test_to_play_losses_vectorized(muzero_config):
    """
    Verify ToPlayLoss and RelativeToPlayLoss vectorized logic.
    """
    device = torch.device("cpu")
    torch.manual_seed(42)
    
    muzero_config.game.num_players = 2
    num_players = 2
    
    # ToPlay is classification
    tp_rep = ClassificationRepresentation(num_players)
    
    tp_loss = ToPlayLoss(muzero_config, device, representation=tp_rep)
    rtp_loss = RelativeToPlayLoss(muzero_config, device, representation=tp_rep)
    
    pipeline = LossPipeline([tp_loss, rtp_loss])
    
    B, T = 2, 4
    
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

def test_ppo_losses_vectorized(ppo_config):
    """
    Verify PPO losses (Policy, Value) vectorized logic.
    """
    device = torch.device("cpu")
    torch.manual_seed(42)
    
    from agents.learner.losses.losses import PPOPolicyLoss, PPOValueLoss
    
    # PPO uses IdentityRepresentation for its internal scaling/math
    pol_rep = ClassificationRepresentation(2) # Actually PPO head is Categorical
    val_rep = IdentityRepresentation() # PPO Value head is Identity/Scalar usually
    
    pol_loss = PPOPolicyLoss(
        ppo_config, device, representation=pol_rep,
        clip_param=0.2, entropy_coefficient=0.01
    )
    val_loss = PPOValueLoss(
        ppo_config, device, representation=val_rep,
        critic_coefficient=0.5
    )
    
    pipeline = LossPipeline([pol_loss, val_loss])
    
    B, T = 2, 1
    num_actions = 2 
    
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
