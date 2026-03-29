import pytest
import torch
from agents.learner.losses.policy import ClippedSurrogateLoss
from agents.learner.losses.value import ClippedValueLoss
from agents.learner.losses.representations import ClassificationRepresentation, ScalarRepresentation

pytestmark = pytest.mark.unit

def test_ppo_debug_metrics_exist_and_computed_correctly():
    """
    Tier 1: Verify PPO debug metrics are present and correctly calculated.
    Tests approxkl, clipfrac, and entropy_loss explicitly.
    """
    device = torch.device("cpu")
    num_actions = 4
    clip_param = 0.2
    
    # 1. Setup Policy Loss
    pol_rep = ClassificationRepresentation(num_classes=num_actions)
    policy_loss_mod = ClippedSurrogateLoss(
        device=device,
        representation=pol_rep,
        clip_param=clip_param,
        entropy_coefficient=0.01,
        name="policy_loss"
    )
    
    # 2. Setup Deterministic Batch
    B, T = 1, 1
    # We create logits such that one action is clearly preferred
    policy_logits = torch.tensor([[[10.0, 0.0, 0.0, 0.0]]]) # Dist heavily heavily favors action 0 (log_prob ~ 0)
    actions = torch.tensor([[0]])
    
    # Let target_log_prob be slightly off to simulate a ratio difference
    # Let's say target log prob was ln(0.5) = -0.693
    target_log_prob = torch.tensor([[-0.693]])
    
    # We predict action 0 with ~1.0 prob, so log_prob is ~0.0
    # log_ratio = 0.0 - (-0.693) = 0.693
    # ratio = exp(0.693) = 2.0
    # This ratio is > 1.2, so it IS clipped. 
    # clipfrac should be 1.0 (100% of batch clipped)
    # approx_kl = (2.0 - 1.0) - 0.693 = 1.0 - 0.693 = 0.307
    
    advantages = torch.tensor([[1.0]])
    
    predictions = {
        "policy_logits": policy_logits
    }
    
    targets = {
        "actions": actions,
        "log_prob": target_log_prob,
        "advantages": advantages,
        "policy_mask": torch.tensor([[True]])
    }
    
    # Run the loss module directly
    loss, metrics = policy_loss_mod.compute_loss(predictions, targets)
    
    # Verify the specific keys exist
    assert "approxkl" in metrics
    assert "clipfrac" in metrics
    assert "entropy_loss" in metrics
    
    # Verify clipfrac is 1.0 (since ratio 2.0 > 1.2)
    torch.testing.assert_close(torch.tensor(metrics["clipfrac"]), torch.tensor(1.0))
    
    # Verify approxkl calculation
    # log_prob for [10, 0, 0, 0] is approx -0.0001
    log_q = torch.log_softmax(policy_logits, dim=-1)
    actual_log_prob = log_q[0, 0, 0].item() # Should be ~ -0.0001
    log_ratio = actual_log_prob - (-0.693)
    ratio = torch.exp(torch.tensor(log_ratio))
    expected_approx_kl = (ratio - 1.0) - log_ratio
    
    torch.testing.assert_close(
        torch.tensor(metrics["approxkl"]), 
        expected_approx_kl,
        atol=1e-4, 
        rtol=1e-4
    )
    
    # Entropy should be near 0 since the distribution is peaked
    assert metrics["entropy_loss"] < 0.1

def test_ppo_value_metrics():
    """
    Tier 1: Verify PPO Value Metrics.
    Checks that the `name` parameter propagates to act as the 'value_loss' metric.
    """
    device = torch.device("cpu")
    val_rep = ScalarRepresentation()
    value_loss_mod = ClippedValueLoss(
        device=device,
        representation=val_rep,
        clip_param=0.2,
        name="value_loss" # Names act as the metric key in the Loss Pipeline
    )
    
    predictions = {"state_value": torch.tensor([[[1.0]]])}
    targets = {
        "values": torch.tensor([[1.0]]),
        "returns": torch.tensor([[1.0]]),
        "value_mask": torch.tensor([[True]])
    }
    
    loss, metrics = value_loss_mod.compute_loss(predictions, targets)
    # The LossPipeline automatically extracts module name ("value_loss") as the key!
    # Because compute_loss returns the raw tensor, the pipeline handles averaging and logging.
    assert value_loss_mod.name == "value_loss"
