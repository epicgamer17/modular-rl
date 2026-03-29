import pytest
import torch
from agents.learner.functional.losses import (
    compute_clipped_surrogate_loss,
    compute_categorical_kl_div,
    compute_mse_loss
)

pytestmark = pytest.mark.unit

def test_compute_clipped_surrogate_loss_analytical():
    """
    Tier 1: Analytical Oracle Test for PPO Clipped Surrogate Objective.
    Verifies clipping logic for both positive and negative advantages.
    """
    # 1. Setup parameters
    clip_param = 0.2 # range [0.8, 1.2]
    ent_coeff = 0.1
    
    # log_probs represent current policy, target_log_probs represent old policy
    # ratio = exp(log_probs - target_log_probs)
    
    # Case 1: Advantage > 0, Ratio > 1+eps (Should clip)
    # log_prob = 0.4, target_log_prob = 0.0 -> ratio = exp(0.4) ~ 1.49 (> 1.2)
    log_probs_1 = torch.tensor([0.4])
    target_log_probs_1 = torch.tensor([0.0])
    adv_1 = torch.tensor([10.0])
    ent_1 = torch.tensor([1.0])
    
    # Expected:
    # ratio = exp(0.4) = 1.4918246976412703
    # surr1 = 1.4918246976412703 * 10 = 14.918246976412703
    # surr2 = 1.2 * 10 = 12.0
    # min(surr1, surr2) = 12.0
    # loss = -12.0 - 0.1 * 1.0 = -12.1
    
    loss_1 = compute_clipped_surrogate_loss(
        log_probs_1, target_log_probs_1, adv_1, clip_param, ent_1, ent_coeff
    )
    torch.testing.assert_close(loss_1, torch.tensor([-12.1]))
    
    # Case 2: Advantage > 0, Ratio < 1-eps (Should NOT clip)
    # log_prob = -0.4, target_log_prob = 0.0 -> ratio = exp(-0.4) ~ 0.67 (< 0.8)
    log_probs_2 = torch.tensor([-0.4])
    target_log_probs_2 = torch.tensor([0.0])
    adv_2 = torch.tensor([10.0])
    ent_2 = torch.tensor([1.0])
    
    # Expected:
    # ratio = exp(-0.4) = 0.6703200460356393
    # surr1 = 6.703200460356393
    # surr2 = 0.8 * 10 = 8.0
    # min(surr1, surr2) = 6.703200460356393
    # loss = -6.703200460356393 - 0.1 = -6.803200460356393
    
    loss_2 = compute_clipped_surrogate_loss(
        log_probs_2, target_log_probs_2, adv_2, clip_param, ent_2, ent_coeff
    )
    torch.testing.assert_close(loss_2, torch.tensor([-6.803200460356393]))

    # Case 3: Advantage < 0, Ratio > 1+eps (Should NOT clip)
    # ratio ~ 1.49, adv = -10.0
    # surr1 = 1.49 * -10 = -14.9
    # surr2 = 1.2 * -10 = -12.0
    # min(-14.9, -12.0) = -14.9 (NOT clipped, we are worsening our performance on a bad action)
    # loss = -(-14.9) - 0.1 = 14.9 - 0.1 = 14.8
    
    loss_3 = compute_clipped_surrogate_loss(
        log_probs_1, target_log_probs_1, -adv_1, clip_param, ent_1, ent_coeff
    )
    # ratio_val = 1.4918246976412703
    # expected = - (ratio_val * -10.0) - 0.1 = 14.918246976412703 - 0.1 = 14.818246976412703
    torch.testing.assert_close(loss_3, torch.tensor([14.818246976412703]))

    # Case 4: Advantage < 0, Ratio < 1-eps (Should clip)
    # ratio ~ 0.67, adv = -10.0
    # surr1 = 0.67 * -10 = -6.7
    # surr2 = 0.8 * -10 = -8.0
    # min(-6.7, -8.0) = -8.0 (CLIPPED to 1-eps)
    # loss = -(-8.0) - 0.1 = 8.0 - 0.1 = 7.9
    
    loss_4 = compute_clipped_surrogate_loss(
        log_probs_2, target_log_probs_2, -adv_2, clip_param, ent_2, ent_coeff
    )
    torch.testing.assert_close(loss_4, torch.tensor([7.9]))

def test_compute_categorical_kl_div():
    """
    Verifies KL divergence calculation between logits and target probabilities.
    """
    # Pred: [0.5, 0.5] -> logits [0, 0]
    # Target: [0.9, 0.1]
    pred_logits = torch.tensor([[0.0, 0.0]])
    target_probs = torch.tensor([[0.9, 0.1]])
    
    # log_p_pred = [-ln(2), -ln(2)] ~ [-0.693, -0.693]
    # KL = sum(p_target * (ln(p_target) - ln(p_pred)))
    # KL = 0.9 * (ln(0.9) - (-ln(2))) + 0.1 * (ln(0.1) - (-ln(2)))
    # KL = 0.9 * (ln(0.9) + ln(2)) + 0.1 * (ln(0.1) + ln(2))
    # KL = 0.9 * ln(1.8) + 0.1 * ln(0.2)
    # KL = 0.9 * 0.5877866649 + 0.1 * (-1.6094379124)
    # KL = 0.5290079984 - 0.1609437912 = 0.3680642072
    
    kl = compute_categorical_kl_div(pred_logits, target_probs)
    torch.testing.assert_close(kl, torch.tensor([0.3680642072]))

def test_compute_mse_loss():
    """
    Simple wrapper for MSE test.
    """
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.5, 1.5])
    # MSE = (0.5^2 + 0.5^2) = 0.25 + 0.25 = 0.5 (if reduction='none', returns per element [0.25, 0.25])
    mse = compute_mse_loss(pred, target, reduction='none')
    torch.testing.assert_close(mse, torch.tensor([0.25, 0.25]))

def test_compute_ppo_value_loss_analytical():
    """
    Tier 1: Analytical Oracle Test for PPO Value Loss Clipping.
    Verifies that the loss correctly identifies the conservative 'max' 
    between unclipped and clipped squared errors.
    """
    from agents.learner.functional.losses import compute_ppo_value_loss
    v_old = torch.tensor([1.0])
    v_target = torch.tensor([2.0])
    clip_param = 0.1 # range [0.9, 1.1]
    
    # Case 1: v_pred is within bounds [0.9, 1.1]
    # v_pred = 1.05
    # unclipped = (1.05 - 2.0)^2 = (-0.95)^2 = 0.9025
    # clipped = 1.0 + 0.05 = 1.05
    # clipped_err = 0.9025
    # loss = 0.5 * 0.9025 = 0.45125
    v_pred_1 = torch.tensor([1.05])
    loss_1 = compute_ppo_value_loss(v_pred_1, v_old, v_target, clip_param)
    torch.testing.assert_close(loss_1, torch.tensor([0.45125]))
    
    # Case 2: v_pred is out of bounds [0.9, 1.1], moving towards target
    # v_pred = 1.4
    # unclipped = (1.4 - 2.0)^2 = 0.36
    # clipped = 1.1
    # clipped_err = (1.1 - 2.0)^2 = 0.81
    # max(0.36, 0.81) = 0.81
    # loss = 0.5 * 0.81 = 0.405
    v_pred_2 = torch.tensor([1.4])
    loss_2 = compute_ppo_value_loss(v_pred_2, v_old, v_target, clip_param)
    torch.testing.assert_close(loss_2, torch.tensor([0.405]))
    
    # Case 3: v_pred is out of bounds, moving AWAY from target
    # v_pred = 0.5
    # unclipped = (0.5 - 2.0)^2 = 2.25
    # clipped = 0.9
    # clipped_err = (0.9 - 2.0)^2 = 1.21
    # max(2.25, 1.21) = 2.25
    # loss = 0.5 * 2.25 = 1.125
    v_pred_3 = torch.tensor([0.5])
    loss_3 = compute_ppo_value_loss(v_pred_3, v_old, v_target, clip_param)
    torch.testing.assert_close(loss_3, torch.tensor([1.125]))
    
    # Case 4: No clipping (clip_param = None)
    # v_pred = 1.4
    # loss = 0.5 * 0.36 = 0.18
    loss_4 = compute_ppo_value_loss(v_pred_2, v_old, v_target, None)
    torch.testing.assert_close(loss_4, torch.tensor([0.18]))

def test_compute_entropy_bonus_analytical():
    """
    Tier 1: Analytical Oracle Test for PPO Entropy Bonus.
    Verifies that the entropy bonus is correctly subtracted from the loss,
    acting as a maximization term (bonus) to encourage exploration.
    """
    clip_param = 0.2
    
    # We fix the surrogate loss part so we can easily isolate the entropy
    log_probs = torch.tensor([0.0])
    target_log_probs = torch.tensor([0.0]) # ratio = 1.0
    adv = torch.tensor([10.0])
    
    # Surrogate without entropy:
    # ratio = 1.0, surr1 = 1.0 * 10 = 10， surr2 = 1.0 * 10 = 10
    # min = 10, loss_base = -10.0
    
    # Case 1: Zero entropy
    ent_1 = torch.tensor([0.0])
    ent_coeff = 0.1
    # Expected: -10.0 - 0.1 * 0.0 = -10.0
    loss_1 = compute_clipped_surrogate_loss(
        log_probs, target_log_probs, adv, clip_param, ent_1, ent_coeff
    )
    torch.testing.assert_close(loss_1, torch.tensor([-10.0]))
    
    # Case 2: Positive entropy (should decrease loss = higher bonus)
    ent_2 = torch.tensor([2.0])
    # Expected: -10.0 - 0.1 * 2.0 = -10.2
    loss_2 = compute_clipped_surrogate_loss(
        log_probs, target_log_probs, adv, clip_param, ent_2, ent_coeff
    )
    torch.testing.assert_close(loss_2, torch.tensor([-10.2]))
    
    # Case 3: Larger entropy coefficient
    ent_coeff_large = 0.5
    # Expected: -10.0 - 0.5 * 2.0 = -11.0
    loss_3 = compute_clipped_surrogate_loss(
        log_probs, target_log_probs, adv, clip_param, ent_2, ent_coeff_large
    )
    torch.testing.assert_close(loss_3, torch.tensor([-11.0]))
