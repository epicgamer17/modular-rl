import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict, Any, Optional


def compute_clipped_surrogate_loss(
    log_probs: Tensor,
    target_log_probs: Tensor,
    advantages: Tensor,
    clip_param: float,
    entropy: Tensor,
    entropy_coefficient: float,
) -> Tensor:
    """
    PPO Clipped Surrogate Loss.

    Args:
        log_probs: current policy log probabilities [B, T]
        target_log_probs: old policy log probabilities [B, T]
        advantages: GAE advantages [B, T]
        clip_param: epsilon clipping parameter
        entropy: policy entropy [B, T]
        entropy_coefficient: scale for entry bonus

    Returns:
        loss tensor [B, T]
    """
    ratio = torch.exp(log_probs - target_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages

    return -torch.min(surr1, surr2) - entropy_coefficient * entropy


def compute_categorical_kl_div(
    pred_logits: Tensor,
    target_probs: Tensor,
    reduction: str = "none",
) -> Tensor:
    """
    KL Divergence between predicted distribution (logits) and target probabilities.
    Requires pred_logits to be raw unnormalized values.

    Returns:
        KL divergence per sample.
    """
    log_prob = F.log_softmax(pred_logits, dim=-1)
    # F.kl_div expects (input_log_prob, target_prob)
    kl = F.kl_div(log_prob, target_probs, reduction=reduction)
    if reduction == "none":
        return kl.sum(dim=-1)
    return kl


# TODO: do we need this wrapper?
def compute_mse_loss(
    predictions: Tensor,
    targets: Tensor,
    reduction: str = "none",
) -> Tensor:
    """
    Mean Squared Error loss wrapper.
    """
    return F.mse_loss(predictions, targets, reduction=reduction)


# TODO: should this be moved somewhere else?
def compute_ppo_value_loss(
    v_pred: Tensor,
    v_old: Tensor,
    v_target: Tensor,
    clip_param: Optional[float] = None,
) -> Tensor:
    """
    PPO Value Loss with optional clipping.

    L = 0.5 * max((v_pred - v_target)^2, (v_clipped - v_target)^2)
    where v_clipped = v_old + clip(v_pred - v_old, -eps, +eps)

    Args:
        v_pred: current value predictions [B, T]
        v_old: values at rollout time [B, T]
        v_target: regression target (e.g. returns) [B, T]
        clip_param: epsilon clipping parameter (None to disable clipping)

    Returns:
        loss tensor [B, T]
    """
    v_err_unclipped = (v_pred - v_target) ** 2

    if clip_param is None:
        return 0.5 * v_err_unclipped

    v_clipped = v_old + torch.clamp(v_pred - v_old, -clip_param, clip_param)
    v_err_clipped = (v_clipped - v_target) ** 2

    # Paper recommends taking max to be conservative
    return 0.5 * torch.max(v_err_unclipped, v_err_clipped)
