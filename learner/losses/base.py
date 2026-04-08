import torch
from typing import Dict, Any, TYPE_CHECKING
from modules.utils import scale_gradient

if TYPE_CHECKING:
    from learner.core import Blackboard

def apply_infrastructure(
    elementwise_loss: torch.Tensor,
    blackboard: 'Blackboard',
    mask_key: str,
) -> torch.Tensor:
    """
    Applies standard infrastructure to a [B, T] elementwise loss:
    1. Gradient Scaling (depth-based)
    2. Weights (Importance Sampling)
    3. Masking
    4. Reduction to Mean Scalar
    """
    # 1. Secure Weights & Gradient Scales from Meta (Truth Source)
    weights = blackboard.meta.get("weights")
    gradient_scales = blackboard.meta.get("gradient_scales")
    masks = blackboard.targets.get(mask_key)

    if weights is None or gradient_scales is None or masks is None:
        # Fallback for simple environments or missing infra
        B, T = elementwise_loss.shape[:2]
        device = elementwise_loss.device
        weights = weights if weights is not None else torch.ones(B, device=device)
        gradient_scales = gradient_scales if gradient_scales is not None else torch.ones((1, T), device=device)
        masks = masks if masks is not None else torch.ones((B, T), device=device, dtype=torch.bool)

    B = weights.shape[0]
    T = gradient_scales.shape[1]

    # Normalize elements
    elementwise_loss = elementwise_loss.reshape(B, T)

    # 2. Scale and Weight
    scaled_loss = scale_gradient(elementwise_loss, gradient_scales)
    weighted_loss = scaled_loss * weights.reshape(B, 1)

    # 3. Mask and Reduce
    masked_weighted_loss = (weighted_loss * masks.float()).sum()
    valid_transition_count = masks.float().sum().clamp(min=1.0)
    
    return masked_weighted_loss / valid_transition_count


