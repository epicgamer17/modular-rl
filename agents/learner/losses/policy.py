import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from agents.learner.losses.base import BaseLoss, LossRepresentation

class PolicyLoss(BaseLoss):
    """Policy prediction loss module."""

    def __init__(
        self,
        device: torch.device,
        representation: LossRepresentation,
        loss_fn: Any,
        loss_factor: float,
        optimizer_name: str = "default",
        mask_key: str = "policy_mask",
        name: Optional[str] = None,
    ):
        super().__init__(
            device=device,
            pred_key="policies",
            target_key="policies",
            mask_key=mask_key,
            representation=representation,
            loss_fn=loss_fn,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
            name=name,
        )

    def compute_loss(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """MuZero Policy Loss: returns (loss, metrics)"""
        loss, metrics = super().compute_loss(predictions, targets)

        with torch.no_grad():
            pred_logits = predictions[self.pred_key]
            target_probs = targets[self.target_key]

            log_q = F.log_softmax(pred_logits, dim=-1)
            log_p = torch.log(target_probs + 1e-10)
            kl = (target_probs * (log_p - log_q)).sum(dim=-1).mean()
            metrics["approx_kl"] = kl.item()

        return loss, metrics

class ClippedSurrogateLoss(BaseLoss):
    def __init__(
        self,
        device: torch.device,
        representation: LossRepresentation,
        clip_param: float,
        entropy_coefficient: float,
        optimizer_name: str = "default",
        name: Optional[str] = None,
    ):
        super().__init__(
            device=device,
            pred_key="policies",
            target_key="actions",
            mask_key="policy_mask",
            representation=representation,
            optimizer_name=optimizer_name,
            name=name,
        )
        self.clip_param = clip_param
        self.entropy_coefficient = entropy_coefficient

    def compute_loss(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """PPO Policy Loss: returns [B, T]"""
        policy_logits = predictions["policies"]
        actions = targets["actions"]
        old_log_probs = targets["old_log_probs"]
        advantages = targets["advantages"]

        # 1. Capture and Validate
        assert (
            policy_logits.ndim == 3
        ), f"ClippedSurrogateLoss requires [B, T, A], got {policy_logits.shape}"
        B, T = old_log_probs.shape[:2]

        # 2. Vectorized Distribution math
        dist = self.representation.to_inference(policy_logits)

        # [B, T]
        log_probs = dist.log_prob(actions)
        assert (
            log_probs.shape == old_log_probs.shape
        ), f"ClippedSurrogateLoss: shape mismatch between log_probs {log_probs.shape} and old_log_probs {old_log_probs.shape}"
        ratio = torch.exp(log_probs - old_log_probs)

        assert (
            ratio.shape == advantages.shape
        ), f"ClippedSurrogateLoss ratio vs advantages: {ratio.shape} vs {advantages.shape}"
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )

        entropy = dist.entropy()

        loss = -torch.min(surr1, surr2) - self.entropy_coefficient * entropy

        # 3. Stats for full sequence
        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean()

        return loss, {"approx_kl": approx_kl.item()}

class ImitationLoss(BaseLoss):
    def __init__(
        self,
        device: torch.device,
        representation: LossRepresentation,
        loss_fn: Any,
        loss_factor: float = 1.0,
        optimizer_name: str = "default",
        mask_key: str = "policy_mask",
        name: Optional[str] = None,
    ):
        super().__init__(
            device=device,
            pred_key="policies",
            target_key="policies",
            mask_key=mask_key,
            representation=representation,
            loss_fn=loss_fn,
            optimizer_name=optimizer_name,
            loss_factor=loss_factor,
            name=name,
        )
